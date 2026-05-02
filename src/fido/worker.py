"""Fido worker — runs one iteration of the work loop for a single repo."""

import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Protocol

import requests as _requests

from fido import hooks, tasks
from fido.claude import ClaudeCode
from fido.config import Config, RepoConfig, RepoMembership, default_sub_dir
from fido.github import GitHub
from fido.issue_cache import IssueNode, IssueTreeCache
from fido.prompts import Prompts, render_active_context
from fido.provider import (
    ContextOverflowError,
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderModel,
    ProviderPressureStatus,
    SessionLeakError,
    TurnSessionMode,
    safe_voice_turn,
    set_thread_kind,
    set_thread_repo,
)
from fido.provider_factory import DefaultProviderFactory
from fido.rocq import ci_task_lifecycle as ci_oracle
from fido.rocq import thread_auto_resolve as thread_resolve_oracle
from fido.state import (
    State,
    _resolve_git_dir,  # pyright: ignore[reportPrivateUsage]
)
from fido.store import FidoStore, PRCommentQueueRecord, ReplyPromiseRecord
from fido.tasks import Tasks
from fido.types import (
    ActiveIssue,
    ActivePR,
    ClosedPR,
    GitIdentity,
    TaskSnapshot,
    TaskStatus,
    TaskType,
)


class GitIdentityError(Exception):
    """Raised when a workspace's git identity doesn't match the authenticated
    GitHub user — fails closed so we never ship commits under the wrong author
    (see #792).
    """


_CI_LOG_TAIL = 200  # max lines of failure log to include in the CI prompt

# Invisible HTML marker appended to pickup comments so future runs can
# distinguish a genuine pickup comment from other things Fido's account may
# have commented on its own issue (fix for #636).
_PICKUP_COMMENT_MARKER = "<!-- fido:pickup -->"

# Invisible HTML marker appended to retry-acknowledgement comments (the ones
# fido posts when restarting on an issue whose prior PR(s) were closed).
# Allows the retry path to dedup across crash-replay cycles (fix for #802).
_RETRY_COMMENT_MARKER = "<!-- fido:retry-ack -->"

# Invisible HTML marker on the one-time comment fido posts when its
# promote-merge guard refuses to mark a PR ready_for_review because
# the branch has no diff vs base (closes #1194).  The PR is left open
# so the human can decide whether more work is needed; the marker
# prevents the worker loop from posting the same comment on every
# subsequent iteration.
_EMPTY_PR_COMMENT_MARKER = "<!-- fido:empty-pr-blocked -->"

log = logging.getLogger(__name__)

_thread_repo: threading.local = threading.local()


_STALE_INDEX_LOCK_SECONDS: float = 30.0
"""How old ``.git/index.lock`` must be before :func:`_remove_stale_index_lock`
treats it as abandoned.  Larger than any normal fido-initiated git
operation, so a concurrent legit writer won't have its lock yanked."""


def _stderr_is_index_lock_error(stderr: str) -> bool:
    """Match git's specific lock-contention error message.

    Git emits ``fatal: Unable to create '<path>/.git/index.lock': File
    exists.`` on any index-touching operation when the lock file is
    already present.  Matching on the stable substring avoids false
    positives from other exit-128 failure modes (closes #827).
    """
    return "Unable to create" in stderr and "index.lock" in stderr


def _remove_stale_index_lock(
    work_dir: Path,
    *,
    stale_after_seconds: float = _STALE_INDEX_LOCK_SECONDS,
    _now: Callable[..., datetime] = datetime.now,
) -> bool:
    """Remove ``<work_dir>/.git/index.lock`` iff it is older than
    *stale_after_seconds*.  Returns ``True`` when a stale lock was
    removed, ``False`` otherwise (lock missing, or too fresh to be
    safely considered abandoned).

    The age gate is deliberately larger than any realistic concurrent
    git write so we never race a legit writer.  Fido is
    single-worker-per-repo, so contention in practice comes from
    provider tool calls that finished seconds ago; 30 s is a safe floor.
    """
    lock = work_dir / ".git" / "index.lock"
    try:
        stat = lock.stat()
    except FileNotFoundError:
        return False
    age = _now(tz=timezone.utc).timestamp() - stat.st_mtime
    if age < stale_after_seconds:
        return False
    lock.unlink()
    return True


class RepoContextFilter(logging.Filter):
    """Inject the current worker thread's repo name into every log record.

    Set ``_thread_repo.repo_name`` before entering a worker loop to tag all
    log records emitted on that thread with the repo's short name.  Records
    from threads that never set the context default to ``"-"``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(getattr(record, "repo_name", None), str):
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

    def has_untriaged(self, repo_name: str) -> bool: ...

    def wait_for_inbox_drain(
        self, repo_name: str, timeout: float | None = None
    ) -> bool: ...

    def note_durable_demand(self, repo_name: str) -> None: ...

    def note_durable_demand_drained(self, repo_name: str) -> None: ...

    def assert_worker_turn_ok(self, repo_name: str) -> None: ...


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
    return default_sub_dir()


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


def build_prompt(
    fido_dir: Path,
    subskill: str,
    context: str,
    labels: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write system and prompt files for a sub-agent session.

    The system file contains ``persona.md`` and ``<subskill>.md`` joined by a
    blank line (matching bash ``printf '%s\\n\\n%s\\n' "$PERSONA" "$skill"``) —
    used by the one-shot ``print_prompt_from_file`` path.

    The skill file contains only ``<subskill>.md`` — used by the persistent
    :class:`~fido.provider.PromptSession` path where the session already has
    ``persona.md`` loaded as system prompt and each turn just needs the
    sub-skill instructions as a user-message preamble.

    The prompt file contains the context string.

    When *labels* contains ``"Blog"``, ``sub/life.md`` is injected between
    persona and skill in both files so the sub-agent has world context for
    blog/journal work (closes #1164).  If ``life.md`` is absent the label is
    silently ignored.

    Returns ``(system_file, prompt_file)`` where both live in *fido_dir*.
    """
    sub = _sub_dir()
    persona = (sub / "persona.md").read_text().rstrip()
    skill = (sub / f"{subskill}.md").read_text().rstrip()

    life: str | None = None
    if "Blog" in (labels or []):
        life_path = sub / "life.md"
        if life_path.exists():
            life = life_path.read_text().rstrip()

    system_file = fido_dir / "system"
    skill_file = fido_dir / "skill"
    prompt_file = fido_dir / "prompt"
    if life:
        system_file.write_text(f"{persona}\n\n{life}\n\n{skill}\n")
        skill_file.write_text(f"{life}\n\n{skill}\n")
    else:
        system_file.write_text(f"{persona}\n\n{skill}\n")
        skill_file.write_text(f"{skill}\n")
    prompt_file.write_text(f"{context}\n")
    return system_file, prompt_file


def _sanitize_status_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse newlines to a single space."""
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _parse_status_nudge(raw: str) -> tuple[str, str]:
    """Extract ``(status, emoji)`` from the JSON nudge response.

    Scans *raw* for the first ``{...}`` object with both fields.  Returns
    ``("", "")`` if parsing fails — callers treat empty fields as "fall back".
    """
    if not raw:
        return "", ""
    candidates = [raw] + [m.group() for m in re.finditer(r"\{.*?\}", raw, re.DOTALL)]
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError, AttributeError:
            continue
        status = obj.get("status") if isinstance(obj, dict) else None
        emoji = obj.get("emoji") if isinstance(obj, dict) else None
        if isinstance(status, str) and isinstance(emoji, str):
            return status, emoji
    return "", ""


def _format_provider_reset_time(resets_at: datetime | None) -> str:
    if resets_at is None:
        return "a little while"
    return resets_at.astimezone(timezone.utc).strftime("%H:%M UTC")


def _provider_pause_activity(status: ProviderPressureStatus) -> str:
    until = _format_provider_reset_time(status.resets_at)
    percent = f"{status.percent_used}%" if status.percent_used is not None else "limit"
    return f"Paused for {status.provider} reset ({percent}, until {until})"


def _provider_pause_status_text(status: ProviderPressureStatus) -> str:
    until = _format_provider_reset_time(status.resets_at)
    return f"Leaving the last 5% for the human until {until}."


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


def provider_start(
    fido_dir: Path,
    *,
    agent: ProviderAgent | None = None,
    model: ProviderModel,
    session: str | None = None,
    timeout: int = 300,
    cwd: Path | str = ".",
    session_mode: TurnSessionMode = TurnSessionMode.REUSE,
) -> str:
    """Start a new provider session from fido_dir/system and fido_dir/prompt.

    When the provider agent already has an attached persistent session, the
    setup prompt is sent through :meth:`ProviderAgent.run_turn` and an empty
    string is returned — there is no subprocess session id to track.

    When no persistent session is attached, a fresh one-shot subprocess is
    spawned via :meth:`ProviderAgent.print_prompt_from_file` and the session id
    is extracted from its raw output.
    """
    del session
    if agent is None:
        raise ValueError("provider_start requires agent")
    if agent.session is not None:
        agent.run_turn(
            _session_turn_prompt(fido_dir),
            model=model,
            retry_on_preempt=True,
            session_mode=session_mode,
        )
        return ""
    system_file = fido_dir / "system"
    prompt_file = fido_dir / "prompt"
    output = agent.print_prompt_from_file(
        system_file, prompt_file, model, timeout, cwd=cwd
    )
    return agent.extract_session_id(output)


def _session_turn_prompt(fido_dir: Path) -> str:
    """Build the user-message body for a persistent provider-session turn.

    The persistent session is constructed with ``sub/persona.md`` as its
    system prompt, so we only need to deliver the sub-skill instructions
    (``fido_dir/skill``) as a user-message preamble — not the full
    ``fido_dir/system`` which would duplicate the persona.  Without this,
    the provider agent saw only the bare context and produced empty output (observed
    as ``setup produced no tasks``).
    """
    skill = (fido_dir / "skill").read_text()
    prompt = (fido_dir / "prompt").read_text()
    return f"{skill}\n\n---\n\n{prompt}"


def provider_run(
    fido_dir: Path,
    *,
    agent: ProviderAgent | None = None,
    model: ProviderModel,
    session: str | None = None,
    timeout: int = 300,
    cwd: Path | str = ".",
    session_mode: TurnSessionMode = TurnSessionMode.REUSE,
    retry_on_preempt: bool = True,
) -> tuple[str, str]:
    """Continue or start a provider session, streaming progress as raw text.

    When the provider agent already has an attached persistent session, the
    prompt is sent through :meth:`ProviderAgent.run_turn` and ``("", "")`` is
    returned.

    When no persistent session is attached, a new one-shot session is started
    from *fido_dir/system* and *fido_dir/prompt*. Returns ``(session_id,
    raw_output)`` where *raw_output* is the full provider output.
    """
    del session
    if agent is None:
        raise ValueError("provider_run requires agent")
    if agent.session is not None:
        agent.run_turn(
            _session_turn_prompt(fido_dir),
            model=model,
            retry_on_preempt=retry_on_preempt,
            session_mode=session_mode,
        )
        return "", ""
    system_file = fido_dir / "system"
    prompt_file = fido_dir / "prompt"
    output = agent.print_prompt_from_file(
        system_file, prompt_file, model, timeout, cwd=cwd
    )
    new_session_id = agent.extract_session_id(output)
    return new_session_id, output


@dataclass(frozen=True)
class PickerChoice:
    """Result of the issue picker.

    *number* is the selected issue; *reason* is a short human-readable
    explanation of how the picker got there (logged on pickup).
    """

    number: int
    title: str
    reason: str


def _has_milestone(issue: dict[str, Any]) -> bool:
    return bool((issue.get("milestone") or {}).get("title"))


def _is_leaked_task_comment(body: str) -> bool:
    """Match top-level PR issue comments fido improvises during a task turn.

    Fix for #669: when a task's work was already done by a prior commit,
    fido sometimes posts a top-level comment like
    ``"BLOCKED: This task is already complete in pushed commit <sha>..."``
    asking a human to mark the task done.  The worker detects these after
    the turn and deletes them so reviewers never see them.

    Narrow by design — only obviously-improvised comments match.  Legitimate
    replies (webhook thread replies, rescope notifications, pickup
    announcements) do not start with ``BLOCKED:`` and do not reference the
    forbidden ``fido task complete`` invocation.
    """
    stripped = (body or "").strip()
    if not stripped:
        return False
    first_line = stripped.splitlines()[0].strip()
    if first_line.startswith("BLOCKED:"):
        return True
    if "cannot run `fido task" in stripped:
        return True
    if "explicitly forbids using `fido task" in stripped:
        return True
    return False


def _issue_assignees(issue: dict[str, Any]) -> list[str]:
    nodes = (issue.get("assignees") or {}).get("nodes") or []
    return [n.get("login", "") for n in nodes if n.get("login")]


def _node_to_dict(
    node: IssueNode,
    cache_index: dict[int, IssueNode],
) -> dict[str, Any]:
    """Render an :class:`IssueNode` in the dict shape the picker helpers
    (``_pick_next_issue`` / ``_walk_to_root`` / ``_descend_issue``) expect.

    The cache stores only open issues, so a sub-issue absent from
    *cache_index* is marked ``state="CLOSED"`` — descent skips those.
    """
    return {
        "number": node.number,
        "title": node.title,
        "createdAt": node.created_at.isoformat(),
        "state": "OPEN",
        "milestone": {"title": node.milestone} if node.milestone else None,
        "assignees": {"nodes": [{"login": login} for login in sorted(node.assignees)]},
        "parent": {"number": node.parent} if node.parent is not None else None,
        "subIssues": {
            "nodes": [
                {
                    "number": child,
                    "state": "OPEN" if child in cache_index else "CLOSED",
                }
                for child in node.sub_issues
            ]
        },
    }


def _pick_next_issue(
    candidates: list[dict[str, Any]],
    login: str,
    *,
    issue_index: dict[int, dict[str, Any]],
) -> PickerChoice | None:
    """Select the next issue to work on from the picker rules in #433.

    *issue_index* maps issue number → full issue dict for every open issue
    in the repo (built from :meth:`~fido.github.GitHub.find_all_open_issues`).
    Issues absent from the index are closed or unplanned and are skipped.

    Algorithm (fix for #775):

    1. For each assigned *candidate*, walk upward via ``.parent`` to the
       true root ancestor — the user requires the chosen issue to be the
       "first open" item from the root down, not just from here.
    2. Dedupe roots by issue number.
    3. Rank roots: milestone-present before milestone-absent, then the
       original creation order of the first assigned descendant.
    4. Descend each root via :func:`_descend_issue` using **strict
       first-priority** semantics — consider only the first open
       sub-issue at each level, abandon the tree if it is blocked by
       another assignee, never claim unassigned children during descent,
       and only return a leaf that *login* already owns (either
       explicitly or via an ancestor in the descent trail).
    """
    roots: list[dict[str, Any]] = []
    seen: set[int] = set()
    order: dict[int, int] = {}
    for idx, candidate in enumerate(candidates):
        root = _walk_to_root(candidate, issue_index=issue_index)
        n = root["number"]
        if n in seen:
            continue
        seen.add(n)
        order[n] = idx
        roots.append(root)

    roots.sort(key=lambda r: (0 if _has_milestone(r) else 1, order[r["number"]]))

    for root in roots:
        choice = _descend_issue(
            root,
            login,
            issue_index=issue_index,
            trail=[root["number"]],
            milestone_source=root if _has_milestone(root) else None,
            owned_on_trail=login in _issue_assignees(root),
        )
        if choice is not None:
            return choice
    return None


def _walk_to_root(
    issue: dict[str, Any],
    *,
    issue_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Walk ``issue.parent`` upward until we hit the root ancestor.

    Uses *issue_index* (a ``{number: issue_dict}`` map built from
    :meth:`~fido.github.GitHub.find_all_open_issues`) to follow parent
    refs without additional API calls.  A parent absent from the index is
    closed or blocked — the walk stops and the current node is treated as
    the root (the closed parent will cause the subtree to be skipped during
    descent).  Returns the root ancestor, or *issue* itself when it has no
    parent.
    """
    current = issue
    visited: set[int] = set()
    while True:
        parent_ref = current.get("parent")
        if not parent_ref or not parent_ref.get("number"):
            return current
        parent_number = parent_ref["number"]
        if parent_number in visited:
            log.warning(
                "picker: parent cycle detected at #%s — stopping walk",
                parent_number,
            )
            return current
        visited.add(parent_number)
        parent = issue_index.get(parent_number)
        if parent is None:
            # Parent is closed or not in the repo's open set — treat current
            # node as root so the descent can proceed from here.
            return current
        current = parent


def _descend_issue(
    issue: dict[str, Any],
    login: str,
    *,
    issue_index: dict[int, dict[str, Any]],
    trail: list[int],
    milestone_source: dict[str, Any] | None,
    owned_on_trail: bool,
) -> PickerChoice | None:
    """Strict first-priority descent (fix for #775).

    At each level, the picker considers **only the first open sub-issue**
    (in the order GitHub returns them — creation order).  If that first
    open child is assigned to someone other than *login*, the whole tree
    is abandoned (return ``None``) — later sibling branches are not
    tried, preserving strict priority semantics.

    Descent never claims unassigned children.  Ownership tracking is
    carried through the recursion via *owned_on_trail*: it stays ``True``
    once any node along the descent path is assigned to *login*.  A leaf
    is returned only when *owned_on_trail* is ``True`` (i.e. Fido owns
    the leaf itself, or an ancestor in ``trail``); otherwise the picker
    walked into a subtree Fido has no stake in, and returns ``None`` so
    the caller can move on to a different candidate's root.
    """
    children = list(issue.get("subIssues", {}).get("nodes") or [])
    # A child is "open" only when it is still present in the index (i.e.
    # still open on GitHub).  Inline state values from the parent query are
    # used as a fast pre-filter; the authoritative check is index membership.
    open_children = [
        issue_index[c["number"]]
        for c in children
        if c.get("state") != "CLOSED" and c.get("number") in issue_index
    ]

    if not open_children:
        if not owned_on_trail:
            log.info(
                "picker: #%s reached as leaf but not owned by %s on trail %s — abandoning",
                issue["number"],
                login,
                "/".join(f"#{n}" for n in trail),
            )
            return None
        depth_note = (
            f" (descended from #{'/#'.join(str(n) for n in trail[:-1])})"
            if len(trail) > 1
            else ""
        )
        milestone_note = ""
        if milestone_source is not None and milestone_source is not issue:
            milestone_note = f", milestone from parent #{milestone_source['number']}"
        return PickerChoice(
            number=issue["number"],
            title=issue.get("title", ""),
            reason=f"picker: pick #{issue['number']}{depth_note}{milestone_note}",
        )

    # Strict first-priority: consider only the first open child.
    first = open_children[0]
    assignees = _issue_assignees(first)
    if assignees and login not in assignees:
        log.info(
            "picker: abandoning tree under #%s — first open sub-issue #%s blocked by %s",
            trail[0],
            first["number"],
            ",".join(assignees),
        )
        return None

    child_owned = login in assignees
    return _descend_issue(
        first,
        login,
        issue_index=issue_index,
        trail=[*trail, first["number"]],
        milestone_source=(
            milestone_source or (first if _has_milestone(first) else None)
        ),
        owned_on_trail=owned_on_trail or child_owned,
    )


def _pick_next_task(task_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-priority eligible task, or ``None``.

    Filters out completed tasks and those prefixed with ``ask:`` or ``defer:``
    (case-insensitive).  Among the remaining candidates the priority order is:

    1. Tasks with ``status`` == ``TaskStatus.IN_PROGRESS`` — resume work
       in flight before scanning for new work.  Without this, a task left
       IN_PROGRESS by an iteration that ended abnormally (subprocess
       crash, fido self-restart) becomes invisible to the picker; combined
       with #989's promote gate (which correctly blocks promote while any
       IN_PROGRESS task exists) the worker deadlocks until the human marks
       the task complete (#999).
    2. Tasks with ``type`` == ``TaskType.CI``
    3. Everything else (first in list wins, including thread tasks)
    """
    eligible = [
        t
        for t in task_list
        if t.get("status") in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
        and not t.get("title", "").lower().startswith("ask:")
        and not t.get("title", "").lower().startswith("defer:")
    ]
    if not eligible:
        return None
    for t in eligible:
        if t.get("status") == TaskStatus.IN_PROGRESS:
            return t
    for t in eligible:
        if t.get("type") == TaskType.CI:
            return t
    return eligible[0]


def _ci_oracle_task_kind(task: Mapping[str, object]) -> ci_oracle.TaskKind:
    title = task.get("title", "")
    title_upper = title.upper() if isinstance(title, str) else ""
    if title_upper.startswith("ASK:"):
        return ci_oracle.TaskAsk()
    if title_upper.startswith("DEFER:"):
        return ci_oracle.TaskDefer()
    if title_upper.startswith("CI FAILURE:") or task.get("type") == TaskType.CI:
        return ci_oracle.TaskCI()
    if task.get("type") == TaskType.THREAD:
        return ci_oracle.TaskThread()
    return ci_oracle.TaskSpec()


def _ci_oracle_task_status(task: Mapping[str, object]) -> ci_oracle.TaskStatus:
    match task.get("status", TaskStatus.PENDING):
        case TaskStatus.COMPLETED | "completed":
            return ci_oracle.StatusCompleted()
        case TaskStatus.BLOCKED | "blocked":
            return ci_oracle.StatusBlocked()
        case _:
            return ci_oracle.StatusPending()


def _ci_oracle_task_rows(
    task_list: Sequence[Mapping[str, object]],
) -> tuple[list[int], dict[int, ci_oracle.TaskRow]]:
    order: list[int] = []
    rows: dict[int, ci_oracle.TaskRow] = {}
    for task_id, task in enumerate(task_list, 1):
        thread = task.get("thread")
        comment_id = thread.get("comment_id") if isinstance(thread, Mapping) else None
        title = task.get("title", "")
        description = task.get("description", "")
        order.append(task_id)
        rows[task_id] = ci_oracle.TaskRow(
            title=title if isinstance(title, str) else "",
            description=description if isinstance(description, str) else "",
            kind=_ci_oracle_task_kind(task),
            status=_ci_oracle_task_status(task),
            source_comment=int(comment_id) if comment_id is not None else None,
        )
    return order, rows


def _ci_oracle_check_key(check_name: str) -> int:
    digest = hashlib.blake2s(check_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") + 1


def _ci_oracle_snapshot(
    check_name: str,
    state: str,
    run_id: str,
) -> ci_oracle.CIFailureSnapshot:
    try:
        run = int(run_id)
    except ValueError:
        run = 1
    if run <= 0:
        run = 1
    conclusion: ci_oracle.CIConclusion = ci_oracle.CIConclusionFailure()
    if state == "TIMED_OUT":
        conclusion = ci_oracle.CIConclusionTimedOut()
    return ci_oracle.CIFailureSnapshot(
        ci_run=run,
        ci_check_name=check_name,
        ci_conclusion=conclusion,
    )


def _assert_ci_failure_matches_oracle(
    task_list: list[dict[str, Any]],
    check_name: str,
    state: str,
    run_id: str,
) -> None:
    order, rows = _ci_oracle_task_rows(task_list)
    new_task = len(order) + 1
    result, created_task = ci_oracle.record_ci_failure(
        _ci_oracle_check_key(check_name),
        _ci_oracle_snapshot(check_name, state, run_id),
        new_task,
        {},
        order,
        rows,
    )
    store_order, next_rows = result
    _store, next_order = store_order
    picked = ci_oracle.pick_next_task(next_order, next_rows)
    if picked != created_task:
        raise AssertionError(
            "ci_task_lifecycle oracle: admitted CI failure was not first pickup"
        )


def _has_pending_asks(task_list: list[dict[str, Any]]) -> bool:
    """Return True if any pending task is an open question (ASK: prefix)."""
    return any(
        t.get("status") == TaskStatus.PENDING
        and t.get("title", "").lower().startswith("ask:")
        for t in task_list
    )


_DECISIVE_REVIEW_STATES = {"APPROVED", "CHANGES_REQUESTED"}
_FRESH_SESSION_NUDGE_ATTEMPT = 4


def _no_commit_nudge(  # pyright: ignore[reportUnusedFunction]
    attempt: int,
    task_title: str,
    task_id: str,
    work_dir: Path | str,
    pr_number: int | None,
) -> str:
    return Prompts("").task_resume_nudge(
        attempt=attempt,
        task_title=task_title,
        task_id=task_id,
        work_dir=str(work_dir),
        pr_number=pr_number,
    )


def _fresh_session_nudge(  # pyright: ignore[reportUnusedFunction]
    task_title: str,
    task_id: str,
    work_dir: Path | str,
    pr_number: int | None,
    branch: str,
) -> str:
    return Prompts("").fresh_session_retry_prompt(
        task_title=task_title,
        task_id=task_id,
        work_dir=str(work_dir),
        pr_number=pr_number,
        branch=branch,
    )


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
    """Extract content between <body>...</body> tags from provider output.

    Returns the extracted content stripped of whitespace, or "" if no body
    tags were found or the raw input was empty.  This enforces the output
    contract in :func:`fido.prompts.rewrite_description_prompt` — the agent is
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
    work_dir: Path,
    gh: Any,
    repo: str,
    pr_number: int,
    issue: int,
    task_list: list[dict[str, Any]],
    existing_body: str = "",
    *,
    agent: ProviderAgent | None = None,
) -> None:
    """Write or rewrite the PR description.

    Handles both initial PR creation (``existing_body=""``; builds work-queue
    section from *task_list*) and post-rescope rewrites (``existing_body``
    contains the current body; preserves the rest section after the ``---``
    divider).

    Generates the description via the provider agent and writes it back via
    ``gh.edit_pr_body``.  The final PATCH is serialized through the same
    ``sync.lock`` that :func:`fido.tasks.sync_tasks` holds, preventing a
    description rewrite from overwriting a concurrent work-queue update.

    Raises ``ValueError`` when the existing body has no ``---`` divider
    (rewrite precondition).  Returns without writing when the agent returns
    empty or un-parseable output — the existing body is kept as-is and a
    warning is logged.
    """
    if agent is None:
        raise ValueError("_write_pr_description requires agent")

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
        # Re-apply the work queue from task_list so a stale PR body snapshot
        # fetched before the Opus call cannot clobber a sync_tasks update that
        # landed while Opus was running (fixes #1013).
        queue = tasks._format_work_queue(  # pyright: ignore[reportPrivateUsage]
            task_list
        )
        rest = tasks._apply_queue_to_body(  # pyright: ignore[reportPrivateUsage]
            rest, queue
        )
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

    prompt = Prompts("").rewrite_description_prompt(existing_body, task_list)
    raw = safe_voice_turn(
        agent, prompt, model=agent.voice_model, log_prefix="_write_pr_description"
    )
    new_desc = _extract_body(raw)
    if not new_desc:
        log.warning(
            "_write_pr_description: skipping PR #%s description update — "
            "no <body> content in provider output (raw=%r)",
            pr_number,
            raw[:200],
        )
        return

    # Ensure "Fixes #N" is always present (the agent preserves it for rewrites via
    # prompt rules; for initial writes we append it here).
    if f"Fixes #{issue}" not in new_desc:
        new_desc = f"{new_desc.rstrip()}\n\nFixes #{issue}."

    new_body = f"{new_desc.strip()}{divider}{rest}"
    # Hold sync.lock during the PATCH so concurrent sync_tasks calls (which
    # also acquire this lock) cannot interleave and overwrite each other.
    with tasks.pr_body_lock(work_dir):
        gh.edit_pr_body(repo, pr_number, new_body)
    log.info("_write_pr_description: PR #%s description written", pr_number)


class AbortHandle:
    """Per-task abort signal.

    Replaces a per-worker :class:`threading.Event` that previously leaked
    across task boundaries (closes #1193): when a rescope marked an
    in-progress task ``completed`` instead of letting :meth:`Worker._cleanup_aborted_task`
    consume the abort signal, the still-set ``Event`` would clobber the
    very next task to enter :meth:`Worker.execute_task`.

    The handle binds an abort request to a specific ``task_id``.  Only
    the cleanup path for the *targeted* task consumes it.  An untargeted
    request (``task_id=None``) is the legacy "abort whatever is running"
    semantic, kept available for the external
    :meth:`WorkerThread.abort_task` entry point that fires before any
    task is in flight.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target_task_id: str | None = None
        self._event = threading.Event()

    def request(self, task_id: str | None) -> None:
        """Request abort of *task_id*.

        ``task_id=None`` is an untargeted request that matches whichever
        task happens to be running.  Real callers (preempt, rescope)
        always know the task they're aborting and should pass it.
        """
        with self._lock:
            self._target_task_id = task_id
            self._event.set()

    def is_active_for(self, task_id: str) -> bool:
        """Return ``True`` when an abort request matches *task_id*.

        An untargeted (legacy) request matches any task.  A targeted
        request matches only its named task — a leaked abort from a
        prior, since-removed task no longer fires here.
        """
        with self._lock:
            if not self._event.is_set():
                return False
            return self._target_task_id is None or self._target_task_id == task_id

    def is_set(self) -> bool:
        """Return whether *any* abort request is pending."""
        return self._event.is_set()

    def clear(self) -> None:
        """Consume the abort request (cleanup path)."""
        with self._lock:
            self._target_task_id = None
            self._event.clear()


class Worker:
    """Fido worker for a single repository.

    Accepts ``work_dir`` and a :class:`~fido.github.GitHub` client via the
    constructor so that tests can inject a mock without patching module-level
    names.  See :ref:`dependency-injection` in CLAUDE.md.
    """

    def __init__(
        self,
        work_dir: Path,
        gh: GitHub,
        abort_task: AbortHandle | None = None,
        repo_name: str = "",
        registry: ActivityReporter | None = None,
        membership: RepoMembership | None = None,
        session: PromptSession | None = None,
        session_issue: int | None = None,
        _tasks: Tasks | None = None,
        provider_agent: ProviderAgent | None = None,
        provider: Provider | None = None,
        prompts: Prompts | None = None,
        config: Config | None = None,
        repo_cfg: RepoConfig | None = None,
        provider_factory: DefaultProviderFactory | None = None,
        first_iteration: bool = False,
        *,
        issue_cache: IssueTreeCache,
    ) -> None:
        self.work_dir = work_dir
        self.gh = gh
        self._abort_task = abort_task if abort_task is not None else AbortHandle()
        self._repo_name = repo_name
        # Replay missed issue_comment webhooks exactly once per WorkerThread
        # lifetime (at startup).  Fix for #794 — without this, top-level PR
        # comments that land while fido is down go unanswered on restart.
        self._first_iteration = first_iteration
        # Per-repo issue tree cache (closes #812).  Required — the
        # picker reads from the cache instead of issuing fresh
        # find_all_open_issues / find_issues GraphQL calls every
        # iteration; webhook events keep the cache in sync.
        self._issue_cache = issue_cache
        self._registry = registry
        self._membership = membership if membership is not None else RepoMembership()
        self._session_issue: int | None = session_issue
        self._next_turn_session_mode = TurnSessionMode.REUSE
        self._tasks = _tasks if _tasks is not None else Tasks(work_dir)
        self._prompts = prompts
        self._config = config
        self._repo_cfg = repo_cfg
        self._provider_factory = (
            DefaultProviderFactory(session_system_file=_sub_dir() / "persona.md")
            if provider_factory is None
            else provider_factory
        )
        self.__dict__["_bootstrap_session"] = session
        if provider is not None:
            self._provider = provider
            if session is not None:
                self._provider.agent.attach_session(session)
        elif provider_agent is not None:
            self._provider = ClaudeCode(agent=provider_agent, session=session)
        elif repo_cfg is not None:
            self._provider = self._provider_factory.create_provider(
                repo_cfg,
                work_dir=work_dir,
                repo_name=repo_name,
                session=session,
            )
        else:
            self._provider = None
        if self._provider is not None:
            self.__dict__["_bootstrap_session"] = self._provider.agent.session

    @property
    def _session(self) -> PromptSession | None:
        if hasattr(self, "_provider") and self._provider is not None:
            return self._provider.agent.session
        return self.__dict__.get("_bootstrap_session")

    @_session.setter
    def _session(self, session: PromptSession | None) -> None:
        if hasattr(self, "_provider") and self._provider is not None:
            self._provider.agent.attach_session(session)
            return
        self.__dict__["_bootstrap_session"] = session

    def _ensure_provider(self) -> Provider:
        """Return the owned provider, creating the configured provider if needed."""
        provider = self._provider
        if provider is None:
            if self._repo_cfg is None:
                raise RuntimeError("worker provider requires explicit repo_cfg")
            provider = self._provider_factory.create_provider(
                self._repo_cfg,
                work_dir=self.work_dir,
                repo_name=self._repo_name,
                session=self._session,
            )
            self._provider = provider
        return provider

    @property
    def _provider_agent(self) -> ProviderAgent:
        provider = self._provider
        if provider is not None:
            return provider.agent
        return self._ensure_provider().agent

    @_provider_agent.setter
    def _provider_agent(self, agent: ProviderAgent) -> None:
        provider = self._provider
        if provider is not None:
            provider.agent.attach_session(agent.session)

    def _get_prompts(self, *, _sub_dir_fn: Callable[..., Path] = _sub_dir) -> Prompts:
        """Return the injected Prompts or build one from the persona file."""
        if self._prompts is not None:
            return self._prompts
        persona_path = _sub_dir_fn() / "persona.md"
        try:
            persona = persona_path.read_text()
        except OSError:
            persona = ""
        return Prompts(persona)

    def resolve_git_dir(
        self, *, _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    ) -> Path:
        """Return the absolute .git directory for self.work_dir."""
        return _resolve_git_dir(self.work_dir, _run=_run)

    def create_context(
        self, *, _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    ) -> WorkerContext:
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
        sync_cmd = f"python -m fido.sync_tasks_cli {self.work_dir} &"
        hooks.add_hooks(self.work_dir, compact_cmd, sync_cmd)
        return compact_cmd, sync_cmd

    def teardown_hooks(self, fido_dir: Path, compact_cmd: str, sync_cmd: str) -> None:
        """Remove hooks and the compact script created by setup_hooks."""
        hooks.remove_hooks(self.work_dir, compact_cmd, sync_cmd)
        (fido_dir / "compact.sh").unlink(missing_ok=True)

    def create_session(self) -> None:
        """Ensure the persistent provider session exists and is on the voice model."""
        self._provider_agent.ensure_session(self._provider_agent.voice_model)

    def stop_session(self) -> None:
        """Stop the persistent provider session, if one exists."""
        self._provider_agent.stop_session()
        self._session = None

    def _consume_turn_session_mode(self) -> TurnSessionMode:
        session_mode = self._next_turn_session_mode
        self._next_turn_session_mode = TurnSessionMode.REUSE
        return session_mode

    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------

    def discover_repo_context(self) -> RepoContext:
        """Discover repo metadata for self.work_dir using the GitHub API.

        The membership (collaborators list) is taken from ``self._membership``,
        which is populated at fido server startup — no per-iteration API
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

    def _issue_has_open_sub_issues(self, repo: str, number: int) -> bool:
        """Return True if issue *number* in *repo* has at least one open
        sub-issue (fix for #780).

        Called each outer iteration so fido abandons an issue that was a
        leaf at pickup but has since acquired children (e.g. fido itself
        groomed it into sub-issues, or a human added children).  Reads
        from the per-repo :class:`IssueTreeCache` (closes #812 follow-up):
        a child is open iff its number is still in the cache index.
        """
        del repo  # repo is the cache's scope already
        cache = self._issue_cache
        index = cache.all_open()
        node = index.get(number)
        if node is None:
            return False
        return any(child in index for child in node.sub_issues)

    def set_status(
        self,
        what: str,
        busy: bool = True,
        *,
        _sub_dir_fn: Callable[..., Path] = _sub_dir,
    ) -> None:
        """Set the authenticated user's GitHub status using provider-generated text.

        Fires a single nudge into the worker's persistent provider session asking
        for a JSON object with both ``status`` and ``emoji`` fields. One
        round-trip instead of the earlier three-to-five one-shot subprocesses
        (closes #505) — no provider spawn overhead, no hang class, and the
        preempt/cancel plumbing handles webhook interleaving cleanly.

        When ``self._session`` is ``None`` (worker has not yet created a
        session), logs and returns — status is best-effort and callers should
        not block on it.
        """
        prompts = self._get_prompts(_sub_dir_fn=_sub_dir_fn)

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

            if self._session is None:
                log.info("set_status: no session available — skipping")
                return

            log.info("set_status: nudging session for status + emoji")
            raw = self._provider_agent.run_turn(
                prompts.status_prompt(activities),
                model=self._provider_agent.voice_model,
                system_prompt=prompts.status_system_prompt(),
            )
            text, emoji = _parse_status_nudge(raw)
            if not text:
                log.warning(
                    "set_status: provider returned no status text — falling back to %r",
                    what[:80],
                )
                text = what[:80]
            text = _sanitize_status_text(text)
            if len(text) > 80:
                text = text[:80]
            if not emoji:
                emoji = ":dog:"

            self.gh.set_user_status(text, emoji, busy=busy)
            log.info("set_status: %s %s", emoji, text)

    def _provider_pressure_status(self) -> ProviderPressureStatus:
        """Return the normalized pressure summary for this worker's provider."""
        return ProviderPressureStatus.from_snapshot(
            self._ensure_provider().api.get_limit_snapshot()
        )

    def _set_provider_pause_status(self, status: ProviderPressureStatus) -> None:
        """Publish a direct break status without consuming more provider budget."""
        what = _provider_pause_activity(status)
        ctx = (
            self._registry.status_update()
            if self._registry is not None
            else nullcontext()
        )
        with ctx:
            if self._registry is not None:
                self._registry.report_activity(self._repo_name, what, False)
            self.gh.set_user_status(
                _provider_pause_status_text(status),
                ":sleeping:",
                busy=False,
            )

    def find_next_issue(self, fido_dir: Path, repo_ctx: RepoContext) -> int | None:
        """Find the next eligible open issue assigned to gh_user.

        Reads candidates + the issue tree from the per-repo
        :class:`~fido.issue_cache.IssueTreeCache` (closes #812) — zero
        GraphQL on the steady-state pick — and verifies the chosen issue
        is still open via a single REST call before committing the
        assignment.
        """
        log.info("finding next eligible issue")
        provider_status = self._provider_pressure_status()
        if provider_status.paused:
            log.info(
                "provider %s is paused at %s%% — leaving the last 5%% for the human",
                provider_status.provider,
                provider_status.percent_used,
            )
            self._set_provider_pause_status(provider_status)
            return None
        choice = self._pick_from_cache(repo_ctx)
        if choice is None:
            log.info(
                "no eligible issues assigned to %s in %s",
                repo_ctx.gh_user,
                repo_ctx.repo,
            )
            self.set_status("All done — no issues to fetch", busy=False)
            return None

        log.info(
            "starting issue #%s: %s (%s)", choice.number, choice.title, choice.reason
        )
        self.gh.add_assignee(repo_ctx.repo, choice.number, repo_ctx.gh_user)
        State(fido_dir).save(
            {
                "issue": choice.number,
                "issue_title": choice.title,
                "issue_started_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        self._ensure_pickup_comment(
            fido_dir, repo_ctx.repo, choice.number, choice.title, repo_ctx.gh_user
        )
        self.set_status(f"Picking up issue #{choice.number}: {choice.title}")
        return choice.number

    def _ensure_pickup_comment(
        self,
        fido_dir: Path,
        repo: str,
        issue: int,
        issue_title: str,
        gh_user: str,
    ) -> None:
        """Ensure the issue has its pickup acknowledgement after restart.

        Fido writes the current issue to durable state before provider-backed
        status/comment generation. If he crashes in that gap, the next run
        resumes the issue instead of re-entering ``find_next_issue()``. A
        separate durable bit lets resume retry this idempotent check.
        """
        state = State(fido_dir)
        if state.load().get("pickup_comment_ensured") is True:
            return
        self.post_pickup_comment(repo, issue, issue_title, gh_user)
        with state.modify() as data:
            data["pickup_comment_ensured"] = True

    def _pick_from_cache(self, repo_ctx: RepoContext) -> PickerChoice | None:
        """Cache-driven picker (closes #812).

        Reads candidates + the full issue tree from the per-repo
        :class:`~fido.issue_cache.IssueTreeCache` — zero GraphQL on the
        steady-state pick — and runs the same priority-rank +
        strict-first-priority descent as the old GraphQL path.

        The cache is bootstrapped eagerly at fido startup (closes #837)
        so it is always loaded by the time this method runs.

        Before returning a non-None choice, makes one cheap REST call
        (``GET /repos/{repo}/issues/{n}``) to confirm the issue is still
        OPEN — guards against the race where a webhook arrives between
        the cache read and the ``add_assignee`` write.
        """
        cache = self._issue_cache
        cache_index = cache.all_open()
        issue_index = {
            n: _node_to_dict(node, cache_index) for n, node in cache_index.items()
        }
        candidates_nodes = cache.assigned_to(repo_ctx.gh_user)
        candidates = [_node_to_dict(n, cache_index) for n in candidates_nodes]
        log.info(
            "picker[cache]: %d open issues in %s, %d assigned to %s",
            len(cache_index),
            repo_ctx.repo,
            len(candidates),
            repo_ctx.gh_user,
        )
        if candidates:
            log.debug(
                "picker[cache]: assigned candidates for %s = %s",
                repo_ctx.gh_user,
                [c["number"] for c in candidates],
            )
        choice = _pick_next_issue(
            candidates,
            repo_ctx.gh_user,
            issue_index=issue_index,
        )
        if choice is None:
            log.info(
                "picker[cache]: no eligible issue for %s in %s",
                repo_ctx.gh_user,
                repo_ctx.repo,
            )
            return None
        log.info(
            "picker[cache]: tentative pick #%s (%s) — running REST verify",
            choice.number,
            choice.reason,
        )
        if not self._verify_cached_issue_still_open(repo_ctx.repo, choice.number):
            log.info(
                "picker[cache]: verify failed for #%s — abandoning pick "
                "(cache will heal on next webhook or hourly reconcile)",
                choice.number,
            )
            return None
        log.info("picker[cache]: verify ok for #%s — committing pick", choice.number)
        return choice

    def _verify_cached_issue_still_open(self, repo: str, number: int) -> bool:
        """One cheap REST call (``GET /repos/{repo}/issues/{n}``) to
        confirm the cached pick is still actionable.  Returns ``True``
        when the issue is still OPEN, ``False`` when CLOSED or missing.
        """
        try:
            data = self.gh.view_issue(repo, number)
        except Exception:
            log.exception(
                "cache verify: view_issue raised for %s#%s — treating as stale",
                repo,
                number,
            )
            return False
        return data.get("state") == "OPEN"

    def _local_branch_exists(self, slug: str) -> bool:
        """Return ``True`` when ``refs/heads/<slug>`` exists locally.

        Discriminates git's exit codes (closes #828):

        - exit 0 → branch exists.
        - exit 1 → branch missing (expected; rev-parse's "no match" code).
        - anything else (128 = stale lock, corrupt repo, bad cwd, …) →
          propagate as :class:`subprocess.CalledProcessError` so callers
          don't silently fall through to the create-branch path on a
          real failure.
        """
        try:
            self._git(["rev-parse", "--verify", "--quiet", f"refs/heads/{slug}"])
            return True
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return False
            raise

    def _git(
        self,
        args: list[str],
        check: bool = True,
        *,
        _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        _now: Callable[..., datetime] = datetime.now,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in self.work_dir.

        Self-heals from stale ``.git/index.lock`` (closes #827): if the
        command fails with git's "Unable to create '.git/index.lock':
        File exists" error AND the lock file is older than
        :data:`_STALE_INDEX_LOCK_SECONDS` seconds, the lock is removed
        and the command is retried exactly once.  The staleness window
        protects against yanking a lock from under a concurrent writer.
        """

        def _call() -> subprocess.CompletedProcess[str]:
            return _run(
                ["git", *args],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                check=check,
            )

        try:
            return _call()
        except subprocess.CalledProcessError as exc:
            if not _stderr_is_index_lock_error(exc.stderr or ""):
                raise
            if not _remove_stale_index_lock(self.work_dir, _now=_now):
                raise
            log.warning(
                "git: removed stale .git/index.lock in %s and retrying %s",
                self.work_dir,
                args[0] if args else "(no args)",
            )
            return _call()

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

    def _reset_local_workspace(
        self, fido_dir: Path, repo_ctx: RepoContext, remote: str
    ) -> None:
        """Reset the workspace to a pristine default-branch state so a fresh
        attempt at the current issue starts from zero (closes #795, #802).

        Supersedes the narrower ``_salvage_interrupted_pr`` (which tried to
        open a PR on a mid-setup orphan branch).  Per user spec: whenever
        no open PR exists for the issue, fido starts fresh — new branch,
        new triage, new task list — and never reuses partial prior work.
        Any crashed-mid-setup tasks.json or orphan branch is garbage.

        Steps:

        1. ``git checkout <default>`` (ignore failure — we may already be
           on default).
        2. ``git fetch origin``.
        3. ``git reset --hard origin/<default>``.
        4. ``git clean -df``.
        5. Delete every non-default local branch (``git branch -D`` each).
           Missing branch is logged, not raised.
        6. Wipe ``tasks.json`` to ``[]``.
        7. Clear ``pr_number`` / ``pr_title`` / ``current_task_id`` from
           ``state.json``.  The issue fields stay so the worker knows what
           it's still working on.
        """
        default = repo_ctx.default_branch
        self._git(["checkout", default], check=False)
        self._git(["fetch", remote])
        self._git(["reset", "--hard", f"{remote}/{default}"])
        self._git(["clean", "-df"])
        # Delete every non-default local branch.
        branches_out = self._git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads/"]
        ).stdout
        if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            branches_out, str
        ):
            for br in branches_out.splitlines():
                br = br.strip()
                if not br or br == default:
                    continue
                result = self._git(["branch", "-D", br], check=False)
                if result.returncode != 0:
                    log.info(
                        "fresh-retry: local branch %s already absent — skipping",
                        br,
                    )
        # Wipe tasks.json — use modify() to get an exclusive flock so the
        # empty-list write can't race with a concurrent task add/complete.
        with self._tasks.modify() as data:
            data.clear()
        # Clear stale PR / task fields from state.json, keep issue fields.
        with State(fido_dir).modify() as state:
            state.pop("pr_number", None)
            state.pop("pr_title", None)
            state.pop("current_task_id", None)
        log.info(
            "fresh-retry: workspace reset to %s/%s (tasks wiped, PR state cleared)",
            remote,
            default,
        )

    def _post_retry_acknowledgement(
        self,
        repo: str,
        issue: int,
        issue_title: str,
        gh_user: str,
        closed_prs: list[int],
    ) -> None:
        """Post a Fido-voiced comment naming the prior closed PRs and
        acknowledging the retry from a clean slate (fix for #802).

        Idempotent on :data:`_RETRY_COMMENT_MARKER`: if a retry-ack is
        already present on the issue since it was last opened (or reopened),
        this is a no-op.  Guards against crash-before-setup-done replays
        spamming the issue with duplicate acknowledgements.
        """
        issue_data = self.gh.view_issue(repo, issue)
        issue_created = issue_data.get("created_at", "")
        events = self.gh.get_issue_events(repo, issue)
        last_opened = issue_created
        for e in events:
            if e.get("event") == "reopened":
                last_opened = e.get("created_at", last_opened)

        comments = self.gh.get_issue_comments(repo, issue)
        has_retry_ack = any(
            c.get("user", {}).get("login") == gh_user
            and c.get("created_at", "") >= last_opened
            and _RETRY_COMMENT_MARKER in c.get("body", "")
            for c in comments
        )
        if has_retry_ack:
            log.info(
                "issue #%s: retry-ack already posted — skipping (closed=%s)",
                issue,
                closed_prs,
            )
            return

        prompts = self._get_prompts()
        prompt = prompts.pickup_retry_comment_prompt(issue_title, closed_prs)
        msg = self._provider_agent.generate_reply(
            prompt, self._provider_agent.voice_model
        )
        if not msg:
            pr_list = ", ".join(f"#{n}" for n in closed_prs)
            msg = (
                f"Retrying — prior PR(s) {pr_list} were closed. "
                f"Starting fresh — new branch, new triage, new task list."
            )
        body = f"{msg}\n\n{_RETRY_COMMENT_MARKER}"
        self.gh.comment_issue(repo, issue, body)
        log.info("posted retry-ack on issue #%s (prior closed: %s)", issue, closed_prs)

    def find_or_create_pr(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        issue: int,
        issue_title: str,
        issue_body: str = "",
        issue_labels: list[str] | None = None,
    ) -> tuple[int, str, bool]:
        """Find or create the branch and draft PR for *issue*.

        Returns ``(pr_number, slug, is_fresh)`` where *is_fresh* is ``True``
        for a newly-created PR and ``False`` for an existing/resumed one.
        Raises ``RuntimeError`` if setup produces no tasks.

        Workflow:
        - **Existing closed PR**: ignore it and create a fresh PR.
        - **Existing open PR**: check out the branch; run the setup sub-agent
          if tasks.json is empty (planning not yet done).
        - **No PR**: generate a slug via the provider brief model, create branch,
          push,
          run setup sub-agent, build the PR body (description + work queue),
          then create the draft PR.
        """
        remote = "origin"
        request = f"{issue_title} (closes #{issue})"

        existing = self.gh.find_pr(repo_ctx.repo, issue, repo_ctx.gh_user)

        if existing is not None:
            pr_number = existing["number"]
            slug = existing["headRefName"]
            pr_title = existing.get("title") or request
            with State(fido_dir).modify() as state:
                state["pr_number"] = pr_number
                state["pr_title"] = pr_title
            # Open PR — resume
            log.info("resuming PR #%s on branch %s", pr_number, slug)
            self._git(["fetch", remote])
            if self._local_branch_exists(slug):
                self._git(["checkout", slug])
            else:
                self._git(["checkout", "-b", slug, "--track", f"{remote}/{slug}"])
            task_list = self._tasks.list()
            if not task_list:
                # Try seeding from PR body first (recovers from state reset)
                self.seed_tasks_from_pr_body(repo_ctx.repo, pr_number)
                task_list = self._tasks.list()
            if not task_list:
                log.info("PR #%s has no tasks — running setup", pr_number)
                pr_body = self.gh.get_pr_body(repo_ctx.repo, pr_number)
                pr_url = f"https://github.com/{repo_ctx.repo}/pull/{pr_number}"
                prior_attempts = self.gh.find_closed_prs_as_context(
                    repo_ctx.repo, issue, repo_ctx.gh_user
                )
                active_ctx = render_active_context(
                    issue=ActiveIssue(number=issue, title=issue_title, body=issue_body),
                    pr=ActivePR(
                        number=pr_number,
                        title=pr_title,
                        url=pr_url,
                        body=pr_body,
                    ),
                    tasks=[],
                    current_task=None,
                    prior_attempts=prior_attempts,
                )
                context = (
                    f"{active_ctx}\n\n"
                    f"Request: {request}\n"
                    f"Repo: {repo_ctx.repo}\n"
                    f"Branch: {slug}\n"
                    f"PR: {pr_number}\n"
                    f"Fork remote: {remote}\n"
                    f"Upstream: {remote}/{repo_ctx.default_branch}\n"
                    f"Work dir: {self.work_dir}"
                )
                build_prompt(fido_dir, "setup", context, labels=issue_labels)
                provider_start(
                    fido_dir,
                    agent=self._provider_agent,
                    model=self._provider_agent.voice_model,
                    cwd=self.work_dir,
                    session=None,
                    session_mode=self._consume_turn_session_mode(),
                )
                if not self._tasks.list():
                    raise RuntimeError(f"setup produced no tasks for PR #{pr_number}")
            log.info(
                "PR: #%s  https://github.com/%s/pull/%s",
                pr_number,
                repo_ctx.repo,
                pr_number,
            )
            return pr_number, slug, False

        # No open PR exists — always take the fresh-retry path (supersedes
        # #795 salvage, which could resurrect a half-planned tasks.json from
        # a setup-midflight crash).  Closes #802.
        #
        # Reset the local workspace to a pristine default-branch state so
        # nothing from a prior attempt (branch, stale tasks, stale PR
        # fields in state.json) leaks into the new one.  Then, if prior
        # closed-not-merged PRs exist, post a Fido-voiced retry-ack
        # comment naming them — idempotent on a marker.
        self._reset_local_workspace(fido_dir, repo_ctx, remote)
        prior_attempts = self.gh.find_closed_prs_as_context(
            repo_ctx.repo, issue, repo_ctx.gh_user
        )
        if prior_attempts:
            self._post_retry_acknowledgement(
                repo_ctx.repo,
                issue,
                issue_title,
                repo_ctx.gh_user,
                [pr.number for pr in prior_attempts],
            )

        # Generate branch slug via the provider brief model
        raw_slug = self._provider_agent.generate_branch_name(
            "Output ONLY a git branch name: 2-4 lowercase words separated by"
            " hyphens, no issue numbers, summarising this request."
            " No explanation, no punctuation, just the branch name."
            f"\n\nRequest: {request}",
            self._provider_agent.brief_model,
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

        # Run setup sub-agent (plans tasks before PR is created)
        log.info("running setup (pre-PR)")
        active_ctx = render_active_context(
            issue=ActiveIssue(number=issue, title=issue_title, body=issue_body),
            pr=None,
            tasks=[],
            current_task=None,
            prior_attempts=prior_attempts,
        )
        context = (
            f"{active_ctx}\n\n"
            f"Request: {request}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Fork remote: {remote}\n"
            f"Upstream: {remote}/{repo_ctx.default_branch}\n"
            f"Work dir: {self.work_dir}"
        )
        build_prompt(fido_dir, "setup", context, labels=issue_labels)
        provider_start(
            fido_dir,
            agent=self._provider_agent,
            model=self._provider_agent.voice_model,
            cwd=self.work_dir,
            session=None,
            session_mode=self._consume_turn_session_mode(),
        )

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
        with State(fido_dir).modify() as state:
            state["pr_number"] = pr_number
            state["pr_title"] = request
        _write_pr_description(
            self.work_dir,
            self.gh,
            repo_ctx.repo,
            pr_number,
            issue,
            self._tasks.list(),
            agent=self._provider_agent,
        )
        task_count = len(
            [t for t in self._tasks.list() if t.get("status") == "pending"]
        )
        log.info("PR: #%s opened with %d tasks", pr_number, task_count)
        log.info("PR: #%s  %s", pr_number, url)

        return pr_number, slug, True

    # ------------------------------------------------------------------
    # Merge conflict handling
    # ------------------------------------------------------------------

    def handle_merge_conflict(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
        issue_labels: list[str] | None = None,
    ) -> bool:
        """Detect and remediate a merge conflict on the branch.

        Returns ``True`` if the branch had merge conflicts and sub-agent was
        invoked to resolve them (the caller should re-run the work loop
        immediately).  Returns ``False`` when there are no conflicts.
        """
        log.info("checking: merge conflicts")
        pr_info = self.gh.get_pr(repo_ctx.repo, pr_number)
        merge_state = pr_info.get("mergeStateStatus", "")
        if merge_state != "DIRTY":
            log.info(
                "merge conflict check skipped — mergeStateStatus=%s",
                merge_state,
            )
            return False

        log.info("merge conflict detected on PR #%s (branch=%s)", pr_number, slug)
        self.set_status(f"Resolving merge conflicts on PR #{pr_number}")

        context = (
            f"PR: {pr_number}\n"
            f"Repo: {repo_ctx.repo}\n"
            f"Branch: {slug}\n"
            f"Upstream: origin/{repo_ctx.default_branch}\n"
            f"Work dir: {self.work_dir}\n"
        )
        build_prompt(fido_dir, "merge", context, labels=issue_labels)
        session_id, _ = provider_run(
            fido_dir,
            agent=self._provider_agent,
            model=self._provider_agent.work_model,
            cwd=self.work_dir,
            session=None,
            session_mode=self._consume_turn_session_mode(),
        )
        log.info("merge conflict resolution done (session=%s)", session_id)
        return True

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
        issue_labels: list[str] | None = None,
    ) -> bool:
        """Check for failing CI checks and run the ci sub-agent to fix them.

        Returns ``True`` if a CI failure was detected and handled (the caller
        should re-run the work loop immediately).  Returns ``False`` when all
        checks are passing or no checks exist.

        On a failure:
        1. Sets the GitHub user status.
        2. Fetches the run failure log (last ``_CI_LOG_TAIL`` lines).
        3. Collects CI-related unresolved review threads.
        4. Builds the ``ci`` sub-agent prompt and runs the provider agent.
        5. Marks the ``CI failure: <check>`` task complete.
        6. Triggers a background sync of the work queue.
        """
        log.info("checking: ci")
        pr_info = self.gh.get_pr(repo_ctx.repo, pr_number)
        merge_state = pr_info.get("mergeStateStatus", "")
        if merge_state not in ("BLOCKED", "DIRTY"):
            log.info(
                "CI check skipped — mergeStateStatus=%s (non-required failures ignored)",
                merge_state,
            )
            return False
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
        _assert_ci_failure_matches_oracle(
            self._tasks.list(),
            check_name,
            failing.get("state", ""),
            run_id,
        )
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
        build_prompt(fido_dir, "ci", context, labels=issue_labels)
        if not self._admit_worker_turn(pr_number):
            return True
        session_id, _ = provider_run(
            fido_dir,
            agent=self._provider_agent,
            model=self._provider_agent.work_model,
            cwd=self.work_dir,
            session=None,
            session_mode=self._consume_turn_session_mode(),
        )
        log.info("CI fix done (session=%s)", session_id)

        # CI failures have no task entry — no complete call needed
        tasks.sync_tasks(self.work_dir, self.gh, blocking=True)
        return True

    def _filter_threads(
        self,
        nodes: list[dict[str, Any]],
        gh_user: str,
        collaborators: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Return unresolved review threads for Python-owned reply handling.

        A thread is included when:
        - it is not resolved,
        - it has at least one comment,
        - the last commenter is not *gh_user* (awaiting a response),
        - the last commenter is either in *collaborators* or ends with ``[bot]``, and
        - the first comment's ID is not claimed or completed in SQLite.

        The last rule prevents the worker reply path from posting a duplicate
        reply to a thread that another webhook or worker attempt already owns.
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
            first_db_id = first_comment.get("databaseId")
            if first_db_id is not None and FidoStore(
                self.work_dir
            ).is_claimed_or_completed(int(first_db_id)):
                continue
            first_login = (first_comment.get("author") or {}).get("login", "")
            result.append(
                {
                    "id": node["id"],
                    "is_bot": last_author.endswith("[bot]"),
                    "first_author": first_login,
                    "first_db_id": first_db_id,
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
        pending_tasks = tasks.thread_tasks_for_auto_resolve_oracle(self._tasks.list())
        config = self._config
        allowed_bots = config.allowed_bots if config is not None else frozenset()
        resolved_any = False
        for node in nodes:
            review_thread = tasks.review_thread_for_auto_resolve_oracle(
                node,
                repo_ctx.gh_user,
                owner=repo_ctx.owner,
                collaborators=repo_ctx.collaborators,
                allowed_bots=allowed_bots,
            )
            decision = thread_resolve_oracle.resolution_decision(
                review_thread,
                pending_tasks,
            )
            if not isinstance(decision, thread_resolve_oracle.ResolveReviewThread):
                if (
                    not review_thread.review_thread_resolved
                    and thread_resolve_oracle.latest_comment_is_fido(review_thread)
                ):
                    log.info(
                        "skipping resolve for thread %s — pending same-thread work "
                        "remains",
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
        """Check for unresolved review threads and handle them via Python.

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

        # Claim each thread's first_db_id before replying.  This makes the
        # claim bidirectional: whichever webhook or worker path prepares the
        # SQLite promise first owns that comment.
        claimable: list[dict[str, Any]] = []
        promises: list[ReplyPromiseRecord] = []
        store = FidoStore(self.work_dir)
        for t in threads:
            first_db_id = t.get("first_db_id")
            if first_db_id is None:
                claimable.append(t)
                continue
            promise = store.prepare_reply(
                owner="worker",
                comment_type="pulls",
                anchor_comment_id=int(first_db_id),
            )
            if promise is None:
                log.info("skipping thread %s — already claimed", first_db_id)
            else:
                promises.append(promise)
                claimable.append(t)

        if not claimable:
            return False

        log.info("unresolved threads: %d", len(claimable))
        from fido import events

        config = self._config
        repo_cfg = self._repo_cfg
        if config is None or repo_cfg is None:
            raise RuntimeError("thread handling requires explicit config and repo_cfg")
        pr_data = self.gh.get_pr(repo_ctx.repo, pr_number)
        pr_title = pr_data.get("title") or ""
        pr_body = pr_data.get("body") or ""
        promise_by_anchor = {
            promise.anchor_comment_id: promise
            for promise in promises
            if promise.anchor_comment_id > 0
        }

        for thread in claimable:
            first_db_id = thread.get("first_db_id")
            if not isinstance(first_db_id, int):
                continue
            promise = promise_by_anchor.get(first_db_id)
            if promise is None:
                continue
            comment = self.gh.get_pull_comment(repo_ctx.repo, first_db_id)
            if comment is None:
                log.info("skipping thread %s — root comment missing", first_db_id)
                store.mark_failed(promise.promise_id)
                continue
            action = events.build_review_comment_action(
                repo_ctx.repo,
                pr_number,
                pr_title,
                pr_body,
                comment,
                comment_body=thread.get("last_body"),
                comment_author=thread.get("last_author"),
            )
            action.context = {
                **(action.context or {}),
                "reply_promise_id": promise.promise_id,
            }
            try:
                category, titles = events.reply_to_comment(
                    action,
                    config,
                    repo_cfg,
                    self.gh,
                    agent=self._provider_agent,
                )
            except Exception:
                store.mark_failed(promise.promise_id)
                raise
            events.queue_reply_tasks(
                category,
                titles,
                config,
                repo_cfg,
                self.gh,
                thread=action.reply_to,
                is_bot=action.is_bot,
                registry=self._registry,
            )
        log.info("threads done")
        tasks.sync_tasks_background(self.work_dir, self.gh)
        return True

    def handle_queued_comments(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
    ) -> bool:
        """Drain durable PR-comment FIFO entries before normal worker turns."""
        del fido_dir, slug
        store = FidoStore(self.work_dir)
        processed_any = False
        while True:
            queued = store.claim_next_pr_comment(
                owner="worker", repo=repo_ctx.repo, pr_number=pr_number
            )
            if queued is None:
                if processed_any and self._registry is not None:
                    self._registry.note_durable_demand_drained(self._repo_name)
                return processed_any
            processed_any = True
            self._handle_queued_comment(store, queued, repo_ctx)

    def _handle_queued_comment(
        self,
        store: FidoStore,
        queued: PRCommentQueueRecord,
        repo_ctx: RepoContext,
    ) -> None:
        from fido import events

        config = self._config
        repo_cfg = self._repo_cfg
        if config is None or repo_cfg is None:
            raise RuntimeError("queued comment handling requires explicit config")
        pr_data = self.gh.get_pr(repo_ctx.repo, queued.pr_number)
        pr_title = pr_data.get("title") or ""
        pr_body = pr_data.get("body") or ""
        promise: ReplyPromiseRecord | None = None
        try:
            action = self._queued_comment_action(
                queued, repo_ctx.repo, pr_title, pr_body
            )
            if action is None:
                store.complete_pr_comment(queued.queue_id)
                return
            promise = store.prepare_reply(
                owner="worker",
                comment_type=queued.comment_type,
                anchor_comment_id=queued.comment_id,
                covered_comment_ids=events.thread_lineage_comment_ids(
                    action.reply_to if queued.comment_type == "pulls" else action.thread
                ),
            )
            if promise is None:
                store.complete_pr_comment(queued.queue_id)
                return
            action.context = {
                **(action.context or {}),
                "reply_promise_id": promise.promise_id,
            }
            category, titles = self._reply_to_queued_comment(
                queued, action, config, repo_cfg
            )
            events.queue_reply_tasks(
                category,
                titles,
                config,
                repo_cfg,
                self.gh,
                thread=action.reply_to
                if queued.comment_type == "pulls"
                else action.thread,
                is_bot=action.is_bot,
                registry=self._registry,
            )
            store.ack_promise(promise.promise_id)
        except Exception as exc:
            if promise is not None:
                store.mark_failed(promise.promise_id)
            store.retry_pr_comment(queued.queue_id, failure_reason=str(exc))
            raise
        store.complete_pr_comment(queued.queue_id)
        tasks.sync_tasks_background(self.work_dir, self.gh)

    def _queued_comment_action(
        self,
        queued: PRCommentQueueRecord,
        repo: str,
        pr_title: str,
        pr_body: str,
    ) -> Any | None:
        if queued.comment_type == "pulls":
            return self._queued_review_comment_action(queued, repo, pr_title, pr_body)
        return self._queued_issue_comment_action(queued, repo, pr_title, pr_body)

    def _reply_to_queued_comment(
        self,
        queued: PRCommentQueueRecord,
        action: Any,
        config: Config,
        repo_cfg: RepoConfig,
    ) -> tuple[str, list[str]]:
        from fido import events

        if queued.comment_type == "pulls":
            return events.reply_to_comment(
                action,
                config,
                repo_cfg,
                self.gh,
                agent=self._provider_agent,
                prompts=self._get_prompts(),
            )
        return events.reply_to_issue_comment(
            action,
            config,
            repo_cfg,
            self.gh,
            agent=self._provider_agent,
            prompts=self._get_prompts(),
        )

    def _queued_review_comment_action(
        self,
        queued: PRCommentQueueRecord,
        repo: str,
        pr_title: str,
        pr_body: str,
    ) -> Any | None:
        from fido import events

        comment = self.gh.get_pull_comment(repo, queued.comment_id)
        if comment is None:
            log.info("queued review comment %s is gone — completing", queued.comment_id)
            return None
        action = events.build_review_comment_action(
            repo,
            queued.pr_number,
            pr_title,
            pr_body,
            comment,
        )
        action.context = {**(action.context or {}), "delivery_id": queued.delivery_id}
        return action

    def _queued_issue_comment_action(
        self,
        queued: PRCommentQueueRecord,
        repo: str,
        pr_title: str,
        pr_body: str,
    ) -> Any | None:
        from fido import events

        comment = self.gh.get_issue_comment(repo, queued.comment_id)
        if comment is None:
            log.info("queued issue comment %s is gone — completing", queued.comment_id)
            return None
        action = events._build_issue_comment_action(  # pyright: ignore[reportPrivateUsage]
            repo,
            queued.pr_number,
            pr_title,
            pr_body,
            comment,
        )
        action.context = {**(action.context or {}), "delivery_id": queued.delivery_id}
        return action

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

    def _push_committed_work_before_yield(
        self, task_title: str, head_before: str, slug: str
    ) -> None:
        """Push any commits that landed during the turn before yielding.

        On a preempt early-return the worker would otherwise leave a
        committed-but-unpushed branch behind: bash command N committed,
        bash command N+1 (``git push``) never ran because the webhook cut
        in.  The next worker iteration may rescope, abort, or otherwise
        forget about the in-flight task and the local commit becomes a
        permanent orphan in the workspace clone (closes #1192).

        Uses ``_commit_provider_leftovers_if_any`` first to absorb any
        uncommitted worktree changes (so the wip-commit safety net still
        runs on a preempt path), then ``ensure_pushed`` to send anything
        new to ``origin``.  Both helpers are idempotent and safe to call
        when nothing changed during the turn.
        """
        head_after = self._commit_provider_leftovers_if_any(task_title, head_before)
        if head_after == head_before:
            return
        self.ensure_pushed("origin", slug)

    def _commit_provider_leftovers_if_any(
        self, task_title: str, head_before: str
    ) -> str:
        """Commit any uncommitted worktree changes left behind by the provider.

        Copilot in particular sometimes applies patches via ``apply_patch``
        but never runs ``git commit`` — the worktree diverges from HEAD and
        the resume loop spins forever waiting for HEAD to move (#654).  If
        HEAD hasn't moved since *head_before* but ``git status`` reports a
        dirty worktree, fido takes ownership and commits everything with a
        ``wip: <task_title> (provider didn't commit)`` message so the outer
        loop makes progress.  Returns the current HEAD sha.

        No-op when HEAD already moved (provider committed on its own) or
        when the worktree is clean (provider produced nothing).
        """
        head_after = self._git(["rev-parse", "HEAD"]).stdout.strip()
        if head_after != head_before:
            return head_after
        porcelain = self._git(["status", "--porcelain"], check=False)
        if porcelain.returncode != 0 or not porcelain.stdout.strip():
            return head_after
        log.warning(
            "task produced uncommitted edits but no commit — committing on "
            "provider's behalf: %s",
            task_title,
        )
        self._git(["add", "-A"])
        self._git(
            [
                "commit",
                "-m",
                f"wip: {task_title} (provider didn't commit)",
            ],
            check=False,
        )
        return self._git(["rev-parse", "HEAD"]).stdout.strip()

    def _snapshot_fido_issue_comment_ids(
        self, repo: str, pr_number: int, fido_login: str
    ) -> set[int]:
        """Snapshot fido-authored issue-comment IDs on a PR (fix for #669).

        Used as the before-image by :meth:`_delete_leaked_task_comments` so
        only comments that appear *during* a task turn are considered for
        cleanup.  Best-effort: on upstream error returns an empty set, which
        conservatively means every later fido comment is treated as new.
        """
        try:
            comments = self.gh.get_issue_comments(repo, pr_number)
        except _requests.RequestException:
            log.exception(
                "leak-check: failed to snapshot issue comments on %s#%d",
                repo,
                pr_number,
            )
            return set()
        return {
            c["id"] for c in comments if c.get("user", {}).get("login") == fido_login
        }

    def _delete_leaked_task_comments(
        self,
        repo: str,
        pr_number: int,
        fido_login: str,
        before_ids: set[int],
    ) -> None:
        """Delete top-level PR issue comments fido improvised during a task
        turn (fix for #669).

        Compares the current set of fido-authored issue comments on
        *pr_number* against *before_ids*; any new comment matching
        :func:`_is_leaked_task_comment` is deleted.  Post-hoc cleanup runs
        after the task completion path; HTTP errors here are logged and
        swallowed so a transient GitHub hiccup doesn't abort the caller.
        """
        try:
            comments = self.gh.get_issue_comments(repo, pr_number)
        except _requests.RequestException:
            log.exception(
                "leak-check: failed to fetch issue comments on %s#%d",
                repo,
                pr_number,
            )
            return
        for c in comments:
            cid = c["id"]
            if cid in before_ids:
                continue
            if c.get("user", {}).get("login") != fido_login:
                continue
            body = c.get("body", "") or ""
            if not _is_leaked_task_comment(body):
                continue
            try:
                self.gh.delete_issue_comment(repo, cid)
                log.warning(
                    "deleted leaked top-level PR comment %d on %s#%d (body=%r)",
                    cid,
                    repo,
                    pr_number,
                    body[:200],
                )
            except _requests.RequestException:
                log.exception(
                    "leak-check: failed to delete comment %d on %s#%d",
                    cid,
                    repo,
                    pr_number,
                )

    def _post_empty_pr_comment_once(self, repo: str, pr_number: int) -> None:
        """Post a one-time BLOCKED comment when the empty-diff guard refuses
        to promote a PR (closes #1194).

        The PR is left open so a human can decide whether more work is
        needed via review comments — fido cannot meaningfully act on an
        empty PR from inside the worker loop.  Idempotent on
        :data:`_EMPTY_PR_COMMENT_MARKER`: if the marker is already
        present in any existing comment on the PR, this is a no-op,
        which prevents the worker loop from re-posting the same comment
        on every iteration.
        """
        try:
            existing = self.gh.get_issue_comments(repo, pr_number)
        except _requests.RequestException:
            log.exception(
                "_post_empty_pr_comment_once: failed to fetch comments on %s#%d",
                repo,
                pr_number,
            )
            return
        if any(_EMPTY_PR_COMMENT_MARKER in (c.get("body") or "") for c in existing):
            return
        body = (
            "BLOCKED: I finished my task list and CI is green, but the branch "
            "has no diff against the base — there's nothing here for a "
            "reviewer to look at. Leaving the PR open so you can decide what "
            "to do: comment with more guidance and I'll pick it up, or close "
            "the PR if there's nothing further.\n\n"
            f"{_EMPTY_PR_COMMENT_MARKER}"
        )
        try:
            self.gh.comment_issue(repo, pr_number, body)
            log.info("posted empty-PR BLOCKED comment on %s#%d", repo, pr_number)
        except _requests.RequestException:
            log.exception(
                "_post_empty_pr_comment_once: failed to post comment on %s#%d",
                repo,
                pr_number,
            )

    def _pr_has_real_diff(self, remote: str, slug: str, default_branch: str) -> bool:
        """Return True iff the branch has any file diff vs the default branch.

        Used as a guard before promoting a draft PR to ready-for-review
        (closes #1194).  A PR whose only commit is the ``wip: start``
        placeholder, or whose branch has no commits ahead of base, has
        nothing to review — promoting it advertises broken state to
        reviewers.

        Compares ``remote/default_branch`` to ``remote/slug`` because
        that's what reviewers actually see; a local commit not yet pushed
        does not count.

        Fails closed: any git error returns ``False`` (treat as no diff,
        do not promote).
        """
        diff = self._git(
            ["diff", "--quiet", f"{remote}/{default_branch}", f"{remote}/{slug}"],
            check=False,
        )
        # `git diff --quiet`: exit 0 = no diff, 1 = diff present.
        if diff.returncode == 0:
            return False
        if diff.returncode == 1:
            return True
        log.warning(
            "_pr_has_real_diff: git diff %s..%s returned %s — treating as no diff",
            default_branch,
            slug,
            diff.returncode,
        )
        return False

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

        Called when ``self._abort_task`` is *active for* this task —
        i.e. an abort was requested with this ``task_id`` (or untargeted)
        and ``execute_task`` observed it mid-execution.  Runs
        ``git_clean`` to restore the working tree, removes the task from
        the queue, clears ``current_task_id`` from state, clears the
        abort handle, and syncs the PR work queue.
        """
        log.info("task aborted: %s", task_title)
        self.git_clean()
        self._tasks.remove(task_id)
        with State(fido_dir).modify() as state:
            state.pop("current_task_id", None)
        self._abort_task.clear()
        tasks.sync_tasks(self.work_dir, self.gh, blocking=True)

    def _yield_for_untriaged(self) -> None:
        """Yield at a provider-turn boundary if untriaged webhooks are waiting.

        Called at both ``provider_run()`` return sites in :meth:`execute_task`
        so that fresh comments and CI events always get a handler turn before
        the next worker turn (#1067).  No-op when the registry is absent
        (standalone tests) or the inbox is already empty.
        """
        if self._registry is None:
            return
        if not self._registry.has_untriaged(self._repo_name):
            return
        log.info(
            "inbox non-empty for %s — yielding turn to untriaged handlers",
            self._repo_name,
        )
        self._registry.wait_for_inbox_drain(self._repo_name, timeout=30.0)

    def _provider_turn_was_preempted(self) -> bool:
        session = self._provider_agent.session
        return bool(session is not None and session.last_turn_cancelled)

    def _has_durable_webhook_demand(self, pr_number: int) -> bool:
        """Return whether durable PR-comment demand should run before work."""
        return FidoStore(self.work_dir).has_pending_pr_comments(
            self._repo_name,
            pr_number=pr_number,
        )

    def _admit_worker_turn(self, pr_number: int) -> bool:
        """Wait until webhook/rescope work drains, then validate turn admission.

        This is the pre-provider gate.  A webhook can arrive after task pickup
        but before ``provider_run()`` starts; in that case the worker must wait
        rather than fire the oracle while the inbox is intentionally non-empty.

        Returns ``False`` when durable PR-comment demand is queued for the
        current PR.  The caller must yield the worker turn instead of starting
        provider work.
        """
        if self._registry is None:
            return True
        if self._registry.has_untriaged(self._repo_name):
            log.info(
                "inbox non-empty for %s before provider turn — waiting",
                self._repo_name,
            )
            self._registry.wait_for_inbox_drain(self._repo_name, timeout=None)
        if self._has_durable_webhook_demand(pr_number):
            log.info(
                "durable webhook demand pending for %s — yielding worker turn",
                self._repo_name,
            )
            self._registry.note_durable_demand(self._repo_name)
            return False
        self._registry.note_durable_demand_drained(self._repo_name)
        self._registry.assert_worker_turn_ok(self._repo_name)
        return True

    def _assert_worker_turn_ok(self) -> None:
        """Fire the handler-preemption oracle before each ``provider_run()``.

        Validates via the extracted ``handler_preemption.v`` FSM that the inbox
        is empty — the worker must not start a new turn while untriaged webhooks
        are pending.  No-op when the registry is absent (standalone tests).
        """
        if self._registry is None:
            return
        self._registry.assert_worker_turn_ok(self._repo_name)

    def _task_still_current(self, fido_dir: Path, task_id: str) -> bool:
        """Return true when *task_id* is still the worker's active task."""
        state_data = State(fido_dir).load()
        if state_data.get("current_task_id") != task_id:
            return False
        current_task_list = self._tasks.list()
        return any(
            t["id"] == task_id
            and t.get("status") not in {TaskStatus.COMPLETED, TaskStatus.BLOCKED}
            for t in current_task_list
        )

    def _report_task_completed_without_commit(
        self,
        repo: str,
        pr_number: int,
        task_id: str,
        task_title: str,
    ) -> None:
        """Explain a completed task that did not produce git progress."""
        log.warning(
            "task %s was marked completed but HEAD did not change; "
            "advancing without branch cleanup",
            task_id,
        )
        prompts = self._get_prompts()
        prompt = prompts.task_completed_without_commit_comment_prompt(task_title)
        msg = self._provider_agent.generate_reply(
            prompt, self._provider_agent.voice_model
        )
        if not msg:
            raise ValueError("task completed without commit comment was empty")
        body = f"{msg}\n\n<!-- fido:task-complete-no-commit -->"
        self.gh.comment_issue(repo, pr_number, body)

    def execute_task(
        self,
        fido_dir: Path,
        repo_ctx: RepoContext,
        pr_number: int,
        slug: str,
        issue_labels: list[str] | None = None,
    ) -> bool:
        """Pick and execute the next pending task via the task sub-agent.

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
        if self._has_durable_webhook_demand(pr_number):
            log.info(
                "durable webhook demand pending for %s — deferring task pickup",
                self._repo_name,
            )
            if self._registry is not None:
                self._registry.note_durable_demand(self._repo_name)
            return True
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
            lineage_ids = thread.get("lineage_comment_ids") or []
            if lineage_ids:
                context_parts.append(
                    "Related thread comment_ids: "
                    + ", ".join(str(comment_id) for comment_id in lineage_ids)
                )
        state_path = fido_dir / "state.json"
        state_data = State(fido_dir).load() if state_path.exists() else {}
        issue_number = state_data.get("issue")
        issue_title = ""
        issue_body = ""
        prior_attempts: list[ClosedPR] = []
        if isinstance(issue_number, int):
            issue_data = self.gh.view_issue(repo_ctx.repo, issue_number)
            issue_title = issue_data.get("title", "")
            issue_body = issue_data.get("body", "")
            prior_attempts = self.gh.find_closed_prs_as_context(
                repo_ctx.repo, issue_number, repo_ctx.gh_user
            )
        else:
            issue_number = None
        pr_data = self.gh.get_pr(repo_ctx.repo, pr_number)
        pr_title = pr_data.get("title", "") or ""
        pr_body = pr_data.get("body", "") or ""
        pr_url = f"https://github.com/{repo_ctx.repo}/pull/{pr_number}"
        active_ctx = render_active_context(
            issue=ActiveIssue(
                number=issue_number if isinstance(issue_number, int) else 0,
                title=issue_title,
                body=issue_body,
            ),
            pr=ActivePR(
                number=pr_number,
                title=pr_title,
                url=pr_url,
                body=pr_body,
            ),
            tasks=[
                TaskSnapshot(
                    title=t.get("title", ""),
                    type=t.get("type", "spec"),
                    status=t.get("status", "pending"),
                    description=t.get("description", ""),
                )
                for t in task_list
            ],
            current_task=TaskSnapshot(
                title=task.get("title", ""),
                type=task.get("type", "spec"),
                status=task.get("status", "pending"),
                description=task.get("description", ""),
            ),
            prior_attempts=prior_attempts,
        )
        context = f"{active_ctx}\n\n" + "\n".join(context_parts)
        build_prompt(fido_dir, "task", context, labels=issue_labels)
        prompts = self._get_prompts()
        head_before = self._git(["rev-parse", "HEAD"]).stdout.strip()
        # Snapshot fido-authored PR comments so we can detect and delete any
        # improvised top-level BLOCKED/leak comments this task turn posts
        # (see #669 and :func:`_is_leaked_task_comment`).
        leak_before_ids = self._snapshot_fido_issue_comment_ids(
            repo_ctx.repo, pr_number, repo_ctx.gh_user
        )
        with State(fido_dir).modify() as state:
            state["current_task_id"] = task["id"]
        self._tasks.update(task["id"], TaskStatus.IN_PROGRESS)
        if not self._admit_worker_turn(pr_number):
            self._tasks.update(task["id"], TaskStatus.PENDING)
            with State(fido_dir).modify() as state:
                state.pop("current_task_id", None)
            return True
        if self._abort_task.is_active_for(task["id"]):
            self._cleanup_aborted_task(fido_dir, task["id"], task_title)
            return True
        if not self._task_still_current(fido_dir, task["id"]):
            log.info(
                "task no longer current after webhook turn admission — restarting loop"
            )
            return True
        session_id, _output = provider_run(
            fido_dir,
            agent=self._provider_agent,
            model=self._provider_agent.work_model,
            cwd=self.work_dir,
            session=None,
            session_mode=self._consume_turn_session_mode(),
            retry_on_preempt=False,
        )
        log.info("task done (session=%s)", session_id)
        if self._provider_turn_was_preempted():
            log.info(
                "task provider turn preempted for %s — yielding to worker loop",
                repo_ctx.repo,
            )
            self._push_committed_work_before_yield(task_title, head_before, slug)
            self._tasks.update(task["id"], TaskStatus.PENDING)
            with State(fido_dir).modify() as state:
                state.pop("current_task_id", None)
            if self._abort_task.is_active_for(task["id"]):
                log.info("consuming abort signal for preempted task %s", task["id"])
                self._abort_task.clear()
            return True
        head_after = self._commit_provider_leftovers_if_any(task_title, head_before)

        if self._abort_task.is_active_for(task["id"]):
            self._cleanup_aborted_task(fido_dir, task["id"], task_title)
            return True

        # Yield before the retry loop so any untriaged webhook handlers get a
        # turn before we start another provider turn (#1067).
        self._yield_for_untriaged()

        # Resume loop: let the provider agent cook until commits appear
        attempt = 0
        fresh_session_retry_used = False
        completed_without_commit = False
        while head_before == head_after:
            current_task_list = self._tasks.list()
            if any(
                t["id"] == task["id"] and t.get("status") == TaskStatus.COMPLETED
                for t in current_task_list
            ):
                self._report_task_completed_without_commit(
                    repo_ctx.repo,
                    pr_number,
                    task["id"],
                    task_title,
                )
                completed_without_commit = True
                break
            attempt += 1
            use_fresh_session = (
                self._provider_agent.supports_no_commit_reset
                and attempt >= _FRESH_SESSION_NUDGE_ATTEMPT
                and not fresh_session_retry_used
            )
            nudge = (
                prompts.fresh_session_retry_prompt(
                    task_title=task_title,
                    task_id=task["id"],
                    work_dir=str(self.work_dir),
                    pr_number=pr_number,
                    branch=slug,
                    issue_number=issue_number,
                    issue_title=issue_title,
                    issue_body=issue_body,
                    pr_title=pr_title,
                    pr_body=pr_body,
                )
                if use_fresh_session
                else prompts.task_resume_nudge(
                    attempt=attempt,
                    task_title=task_title,
                    task_id=task["id"],
                    work_dir=str(self.work_dir),
                    pr_number=pr_number,
                )
            )
            (fido_dir / "prompt").write_text(nudge)
            if use_fresh_session:
                fresh_session_retry_used = True
                log.info(
                    "task produced no commits — retrying with fresh session (attempt %d)",
                    attempt,
                )
            else:
                log.info(
                    "task produced no commits — nudging session (attempt %d)",
                    attempt,
                )
            pending_session_mode = self._consume_turn_session_mode()
            session_mode = (
                TurnSessionMode.FRESH
                if pending_session_mode == TurnSessionMode.FRESH or use_fresh_session
                else TurnSessionMode.REUSE
            )
            if not self._admit_worker_turn(pr_number):
                return True
            if self._abort_task.is_active_for(task["id"]):
                self._cleanup_aborted_task(fido_dir, task["id"], task_title)
                return True
            if not self._task_still_current(fido_dir, task["id"]):
                log.info(
                    "task no longer current after webhook turn admission — "
                    "stopping retry"
                )
                return True
            session_id, _output = provider_run(
                fido_dir,
                agent=self._provider_agent,
                model=self._provider_agent.work_model,
                cwd=self.work_dir,
                session=session_id or None,
                session_mode=session_mode,
                retry_on_preempt=False,
            )
            log.info("task resume done (session=%s)", session_id)
            if self._provider_turn_was_preempted():
                log.info(
                    "task provider resume preempted for %s — yielding to worker loop",
                    repo_ctx.repo,
                )
                self._push_committed_work_before_yield(task_title, head_before, slug)
                self._tasks.update(task["id"], TaskStatus.PENDING)
                with State(fido_dir).modify() as state:
                    state.pop("current_task_id", None)
                if self._abort_task.is_active_for(task["id"]):
                    log.info("consuming abort signal for preempted task %s", task["id"])
                    self._abort_task.clear()
                return True
            head_after = self._commit_provider_leftovers_if_any(task_title, head_before)

            if self._abort_task.is_active_for(task["id"]):
                self._cleanup_aborted_task(fido_dir, task["id"], task_title)
                return True

            # Yield at the retry boundary so untriaged handlers get their turn
            # before we loop back for another provider run (#1067).
            self._yield_for_untriaged()

        if completed_without_commit:
            config = self._config
            allowed_bots = config.allowed_bots if config is not None else frozenset()
            self._tasks.complete_with_resolve(
                task["id"],
                self.gh,
                collaborators=repo_ctx.collaborators,
                allowed_bots=allowed_bots,
            )
            with State(fido_dir).modify() as state:
                state.pop("current_task_id", None)
            tasks.sync_tasks(self.work_dir, self.gh, blocking=True)
            self._delete_leaked_task_comments(
                repo_ctx.repo, pr_number, repo_ctx.gh_user, leak_before_ids
            )
            return True

        self._squash_wip_commit("origin", slug, repo_ctx.default_branch)
        pushed = self.ensure_pushed("origin", slug)
        if pushed is not False:
            config = self._config
            allowed_bots = config.allowed_bots if config is not None else frozenset()
            self._tasks.complete_with_resolve(
                task["id"],
                self.gh,
                collaborators=repo_ctx.collaborators,
                allowed_bots=allowed_bots,
            )
            with State(fido_dir).modify() as state:
                state.pop("current_task_id", None)
            tasks.sync_tasks(self.work_dir, self.gh, blocking=True)
        # Sweep any leaked top-level PR comments (BLOCKED: ...) the provider
        # improvised during this task turn.  Runs after task completion so a
        # transient GitHub error during cleanup doesn't block progress.
        self._delete_leaked_task_comments(
            repo_ctx.repo, pr_number, repo_ctx.gh_user, leak_before_ids
        )
        return True

    def seed_tasks_from_pr_body(self, repo: str, pr_number: int) -> None:
        """Seed tasks.json from the PR body work-queue markers if tasks.json is empty.

        Extracts **both** unchecked (``- [ ] ...``) and checked
        (``- [x] ...``) task items between ``WORK_QUEUE_START`` and
        ``WORK_QUEUE_END`` markers.  Unchecked items are added as pending;
        checked items are added with status=COMPLETED so the downstream
        "all tasks done → promote the PR" logic can see them (fix for #646 —
        previously completed items were skipped, and a PR whose work queue
        held only completed items looked like a fresh PR needing setup).

        Lines without a ``<!-- type:X -->`` comment are skipped with a
        warning (e.g. stale multi-line task bodies from older PR bodies).

        No-op if tasks.json is already non-empty, or if no markers /
        items are found.
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
        parsed: list[tuple[str, TaskType, TaskStatus]] = []
        for line in match.group(1).splitlines():
            check = re.match(r"^- \[( |x)\] (.+)$", line)
            if not check:
                continue
            status = (
                TaskStatus.COMPLETED if check.group(1) == "x" else TaskStatus.PENDING
            )
            rest = check.group(2)
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
            parsed.append((title, task_type, status))
        if not parsed:
            return
        pending = 0
        completed = 0
        for title, task_type, status in parsed:
            self._tasks.add(title, task_type, status=status)
            if status == TaskStatus.COMPLETED:
                completed += 1
            else:
                pending += 1
        log.info(
            "seeded %d tasks from PR body (%d pending, %d completed)",
            len(parsed),
            pending,
            completed,
        )

    def post_pickup_comment(
        self, repo: str, issue: int, issue_title: str, gh_user: str
    ) -> None:
        """Post a Fido-flavoured pickup comment on the issue if not already posted.

        Detects a prior pickup comment by looking for
        :data:`_PICKUP_COMMENT_MARKER` in a gh_user comment created after the
        issue was last opened (handles reopened issues). Without the marker
        we can't tell a pickup comment apart from other things Fido's account
        may have said on its own issue (fix for #636).
        """
        issue_data = self.gh.view_issue(repo, issue)
        issue_created = issue_data.get("created_at", "")
        events = self.gh.get_issue_events(repo, issue)
        last_opened = issue_created
        for e in events:
            if e.get("event") == "reopened":
                last_opened = e.get("created_at", last_opened)

        comments = self.gh.get_issue_comments(repo, issue)
        has_pickup_comment = any(
            c.get("user", {}).get("login") == gh_user
            and c.get("created_at", "") >= last_opened
            and _PICKUP_COMMENT_MARKER in c.get("body", "")
            for c in comments
        )
        if has_pickup_comment:
            log.info("issue #%s: pickup comment already exists — skipping", issue)
            return

        prompts = self._get_prompts()
        prompt = prompts.pickup_comment_prompt(issue_title)
        msg = self._provider_agent.generate_reply(
            prompt, self._provider_agent.voice_model
        )
        if not msg:
            msg = f"Picking up issue: {issue_title}"

        body = f"{msg}\n\n{_PICKUP_COMMENT_MARKER}"
        self.gh.comment_issue(repo, issue, body)
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
        # In-flight work (in_progress) blocks promote/merge as much as
        # pending work — a task the worker is actively executing right
        # now is unfinished, even though its status is not "pending".
        # Without this, fido marks the PR ready (and runs the body
        # checkbox sync) before the trailing task finishes (#988).
        pending = [
            t
            for t in task_list
            if t.get("status") in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
        ]

        # Merge only if: approved + not draft + no incomplete tasks
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
            checks = self.gh.pr_checks(repo_ctx.repo, pr_number)
            required = self.gh.get_required_checks(
                repo_ctx.repo, repo_ctx.default_branch
            )
            if not ci_ready_for_review(checks, required):
                log.info(
                    "PR #%s: changes requested — all addressed, but CI not yet passing — deferring re-request",
                    pr_number,
                )
                return 0
            promoted_from_draft = False
            if is_draft:
                if not self._pr_has_real_diff("origin", slug, repo_ctx.default_branch):
                    log.warning(
                        "PR #%s: refusing ready_for_review — branch has no diff "
                        "vs %s (#1194)",
                        pr_number,
                        repo_ctx.default_branch,
                    )
                    self._post_empty_pr_comment_once(repo_ctx.repo, pr_number)
                    return 0
                log.info(
                    "PR #%s: changes requested — all addressed, CI passing — marking ready",
                    pr_number,
                )
                self.gh.pr_ready(repo_ctx.repo, pr_number)
                promoted_from_draft = True
            missing = sorted(repo_ctx.collaborators - set(requested_reviewers))
            if missing:
                log.info(
                    "PR #%s: changes requested — all addressed, CI passing — re-requesting review from %s",
                    pr_number,
                    ", ".join(missing),
                )
                self.gh.add_pr_reviewers(repo_ctx.repo, pr_number, missing)
            return 1 if promoted_from_draft else 0

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
                    "PR #%s: %d tasks still pending or in-progress — not promoting yet",
                    pr_number,
                    len(pending),
                )
                return 0
            checks = self.gh.pr_checks(repo_ctx.repo, pr_number)
            required = self.gh.get_required_checks(
                repo_ctx.repo, repo_ctx.default_branch
            )
            if not ci_ready_for_review(checks, required):
                log.info(
                    "PR #%s: work complete but CI not yet green — deferring promote",
                    pr_number,
                )
                return 0
            thread_nodes = self.gh.get_review_threads(
                repo_ctx.owner, repo_ctx.repo_name, pr_number
            )
            if any(not n["isResolved"] for n in thread_nodes):
                log.info(
                    "PR #%s: work complete but unresolved review threads remain — deferring promote",
                    pr_number,
                )
                return 0
            if not self._pr_has_real_diff("origin", slug, repo_ctx.default_branch):
                log.warning(
                    "PR #%s: refusing ready_for_review — branch has no diff "
                    "vs %s (#1194)",
                    pr_number,
                    repo_ctx.default_branch,
                )
                self._post_empty_pr_comment_once(repo_ctx.repo, pr_number)
                return 0
            log.info("PR #%s: work complete, CI green — marking ready", pr_number)
            self.gh.pr_ready(repo_ctx.repo, pr_number)
            missing = sorted(repo_ctx.collaborators - set(requested_reviewers))
            if missing:
                log.info(
                    "PR #%s: requesting review from %s",
                    pr_number,
                    ", ".join(missing),
                )
                self.gh.add_pr_reviewers(repo_ctx.repo, pr_number, missing)
            self._enable_auto_merge_on_approval(repo_ctx, pr_number)
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
                self._enable_auto_merge_on_approval(repo_ctx, pr_number)
            else:
                log.info(
                    "PR #%s: CI not yet passing — waiting before requesting review",
                    pr_number,
                )
            return 0

        log.info("PR #%s: no work", pr_number)
        self.set_status("Napping — waiting for work", busy=False)
        return 0

    def _enable_auto_merge_on_approval(
        self, repo_ctx: RepoContext, pr_number: int
    ) -> None:
        """Try to enable GitHub's native auto-merge on a just-ready / review-
        requested PR so the merge lands the instant approval arrives — rather
        than relying on the worker's 60s polling cadence to notice the state
        flip (fix for #787).

        Silently no-ops when the repo has auto-merge disabled (common on
        repos without branch protection).  The worker's approval-polling
        path remains as a fallback so this change is strictly additive.
        """
        try:
            enabled = self.gh.try_enable_auto_merge(
                repo_ctx.repo, pr_number, squash=True
            )
        except Exception as exc:
            log.warning(
                "PR #%s: failed to enable auto-merge (%s) — will rely on polling",
                pr_number,
                exc,
            )
            return
        if enabled:
            log.info(
                "PR #%s: auto-merge enabled — will merge on approval",
                pr_number,
            )
        else:
            log.info(
                "PR #%s: auto-merge not available on this repo — "
                "will rely on polling to merge after approval",
                pr_number,
            )

    def rescope_before_pick(self) -> None:
        """Run a synchronous provider-agent rescope before picking the next task.

        Called at the start of every worker iteration so the PR task list
        stays fresh.  Skips when :attr:`_config` or :attr:`_repo_cfg` are not
        injected (standalone :func:`run` invocation) or when fewer than two
        tasks are pending (nothing to reorder).

        Uses the same ``_on_changes`` and ``_on_done`` callbacks as the
        background rescope triggered by ``create_task()``: thread-task authors
        are notified of any changes and the PR description is rewritten after a
        successful reorder.

        Does **not** pass ``_on_inprogress_affected``: there is no running task
        to abort at pick time, so the abort signal would be either a no-op or
        harmful to the task that is about to be picked.
        """
        if self._config is None or self._repo_cfg is None:
            log.debug("rescope_before_pick: no config/repo_cfg — skipping")
            return

        pending = [
            t for t in self._tasks.list() if t.get("status") == TaskStatus.PENDING
        ]
        if len(pending) < 2:
            log.debug("rescope_before_pick: fewer than 2 pending tasks — skipping")
            return

        from fido.events import (
            _get_commit_summary,  # pyright: ignore[reportPrivateUsage]
            _make_reorder_kwargs,  # pyright: ignore[reportPrivateUsage]
            _rewrite_pr_description,  # pyright: ignore[reportPrivateUsage]
        )
        from fido.tasks import reorder_tasks

        commit_summary = _get_commit_summary(self.work_dir)
        kwargs = _make_reorder_kwargs(
            self.work_dir,
            self._config,
            self._repo_cfg,
            None,  # no _on_inprogress_affected: no running task to abort at pick time
            self.gh,
            self._provider_agent,
            self._get_prompts(),
            _rewrite_pr_description,
        )
        log.info("rescope_before_pick: rescoping task list before next pick")
        reorder_tasks(self.work_dir, commit_summary, **kwargs)

    def assert_git_identity(self, *, phase: str) -> None:
        """Enforce the git-identity invariant (see #792).

        An invariant, checked at both the start (pre-condition) and end
        (post-condition) of every worker iteration: the workspace's configured
        git identity must match the authenticated GitHub user.  If it doesn't,
        raise — loud failure in the crash log beats silently shipping a commit
        under the wrong author.

        The expected identity is derived dynamically from the GitHub API —
        ``name`` from the account's display name (fallback: login), ``email``
        from the noreply form ``{id}+{login}@users.noreply.github.com`` so a
        real address is never used.

        A post-condition failure (``phase="post"``) is the scary case: it means
        something *during the iteration* mutated the config.  Either way the
        worker aborts and the watchdog restarts.

        Refreshes the gh-CLI token before the comparison so a host-side
        ``gh auth switch`` (rare, but happens during ops work) is picked
        up automatically rather than crash-looping the worker until a
        process restart (closes #1207).
        """
        self.gh.refresh_token()
        expected = self.gh.get_authenticated_identity()
        actual = GitIdentity(
            name=self._git_config_get("user.name"),
            email=self._git_config_get("user.email"),
        )
        if actual != expected:
            raise GitIdentityError(
                f"git identity invariant violated ({phase}) in {self.work_dir}: "
                f"actual={actual} expected={expected}"
            )

    def _git_config_get(self, key: str) -> str:
        """Return ``git config --get <key>`` in the workspace, empty on unset."""
        result = subprocess.run(
            ["git", "config", "--get", key],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

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

            self.assert_git_identity(phase="pre")
            try:
                return self._run_iteration(ctx)
            finally:
                self.assert_git_identity(phase="post")

    def _run_iteration(self, ctx: WorkerContext) -> int:
        """Body of :meth:`run` — guaranteed to run between the pre- and
        post-condition identity checks in :meth:`run`.
        """
        repo_ctx = self.discover_repo_context()
        log.info(
            "repo=%s user=%s default_branch=%s",
            repo_ctx.repo,
            repo_ctx.gh_user,
            repo_ctx.default_branch,
        )

        compact_cmd, sync_cmd = self.setup_hooks(ctx.fido_dir)
        session_fresh = self._session is None
        if session_fresh:
            self.create_session()
        try:
            issue = self.get_current_issue(ctx.fido_dir, repo_ctx.repo)
            if issue is None:
                issue = self.find_next_issue(ctx.fido_dir, repo_ctx)
            if issue is None:
                return 0

            # Guard: if the active issue has acquired open sub-issues
            # post-pickup (e.g. fido groomed it into sub-issues, or a
            # human added children), it is no longer a leaf.  Drop the
            # issue out of state.json so the next iteration's picker
            # re-descends from the true root and lands on the first
            # open child.  Assignees are not touched (per-repo policy).
            # Fix for #780.
            if self._issue_has_open_sub_issues(repo_ctx.repo, issue):
                log.warning(
                    "issue #%s has acquired open sub-issues — abandoning "
                    "state so picker re-descends next iteration (see #780)",
                    issue,
                )
                with State(ctx.fido_dir).modify() as state:
                    state.pop("issue", None)
                    state.pop("issue_title", None)
                    state.pop("issue_started_at", None)
                    state.pop("pr_number", None)
                    state.pop("pr_title", None)
                    state.pop("current_task_id", None)
                return 0

            self._next_turn_session_mode = TurnSessionMode.REUSE
            if not session_fresh and issue != self._session_issue:
                log.info(
                    "worker: new issue #%s — restarting session at issue boundary",
                    issue,
                )
                self._next_turn_session_mode = TurnSessionMode.FRESH
            self._session_issue = issue

            issue_data = self.gh.view_issue(repo_ctx.repo, issue)
            issue_title = issue_data["title"]
            issue_body = issue_data.get("body", "") or ""
            issue_labels = issue_data.get("labels", [])
            self._ensure_pickup_comment(
                ctx.fido_dir, repo_ctx.repo, issue, issue_title, repo_ctx.gh_user
            )
            pr_number, slug, pr_is_fresh = self.find_or_create_pr(
                ctx.fido_dir,
                repo_ctx,
                issue,
                issue_title,
                issue_body,
                issue_labels=issue_labels,
            )
            if self._first_iteration:
                recovered_comments = FidoStore(
                    self.work_dir
                ).recover_in_progress_pr_comments(repo=repo_ctx.repo)
                if recovered_comments:
                    log.warning(
                        "recovered %s in-progress PR comment claim(s) for %s",
                        len(recovered_comments),
                        repo_ctx.repo,
                    )
            recovery_provider = (
                self._ensure_provider().provider_id
                if self._repo_cfg is None
                else self._repo_cfg.provider
            )
            recovery_repo_cfg = RepoConfig(
                name=repo_ctx.repo,
                work_dir=self.work_dir,
                provider=recovery_provider,
                membership=repo_ctx.membership,
            )
            recovery_config = Config(
                port=0,
                secret=b"",
                repos={repo_ctx.repo: recovery_repo_cfg},
                allowed_bots=frozenset(),
                log_level="WARNING",
                sub_dir=_sub_dir(),
            )
            from fido.events import recover_reply_promises

            recovered_promises = recover_reply_promises(
                ctx.fido_dir,
                recovery_config,
                recovery_repo_cfg,
                self.gh,
                pr_number,
                agent=self._provider_agent,
                prompts=self._get_prompts(),
            )
            self.seed_tasks_from_pr_body(repo_ctx.repo, pr_number)
            if self._first_iteration and not pr_is_fresh:
                # One-shot replay of missed issue_comment webhooks (fix #794).
                # Runs only on the first iteration per WorkerThread lifetime so
                # the steady-state loop stays fast; create_task dedups on
                # comment_id so re-tasking already-handled comments is a no-op.
                from fido.events import backfill_missed_pr_comments

                backfill_missed_pr_comments(
                    recovery_config,
                    recovery_repo_cfg,
                    self.gh,
                    pr_number,
                    gh_user=repo_ctx.gh_user,
                )
                self._first_iteration = False
            if pr_is_fresh:
                log.info("fresh PR — skipping CI/thread/rescope checks")
            else:
                self.rescope_before_pick()
                if self.handle_merge_conflict(
                    ctx.fido_dir,
                    repo_ctx,
                    pr_number,
                    slug,
                    issue_labels=issue_labels,
                ):
                    return 1
                if self.handle_queued_comments(ctx.fido_dir, repo_ctx, pr_number, slug):
                    return 1
                if self.handle_ci(
                    ctx.fido_dir,
                    repo_ctx,
                    pr_number,
                    slug,
                    issue_labels=issue_labels,
                ):
                    return 1
                if self.handle_threads(ctx.fido_dir, repo_ctx, pr_number, slug):
                    return 1
            if self.execute_task(
                ctx.fido_dir,
                repo_ctx,
                pr_number,
                slug,
                issue_labels=issue_labels,
            ):
                self.resolve_addressed_threads(repo_ctx, pr_number)
                return 1
            promote_result = self.handle_promote_merge(
                ctx.fido_dir, repo_ctx, pr_number, slug, issue
            )
            return 1 if recovered_promises and promote_result == 0 else promote_result
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

    **Session persistence**

    ``_provider`` and ``_session_issue`` survive individual :class:`Worker`
    crashes: each new ``Worker`` receives the existing provider via the
    constructor and hands it back after ``run()`` returns (even on exception).
    The provider owns the persistent session object, while ``_session`` remains
    a compatibility shim over that attached session for existing worker code.
    When this thread itself crashes, :class:`~fido.registry.WorkerRegistry`
    rescues the live provider from the dead thread and passes it to the
    replacement thread via the *provider* constructor parameter, so both the
    provider instance and its attached session persist across Worker-level and
    WorkerThread-level crashes.

    Neither ``_provider`` nor ``_session_issue`` survive a full fido restart
    — ``os.execvp`` replaces the process, so a new ``WorkerThread`` starts
    with no provider-attached session and creates a fresh session on its first
    iteration.
    """

    def __init__(
        self,
        work_dir: Path,
        repo_name: str,
        gh: GitHub,
        registry: ActivityReporter | None = None,
        membership: RepoMembership | None = None,
        session: PromptSession | None = None,
        session_issue: int | None = None,
        provider: Provider | None = None,
        config: Config | None = None,
        repo_cfg: RepoConfig | None = None,
        provider_factory: DefaultProviderFactory | None = None,
        *,
        issue_cache: IssueTreeCache,
    ) -> None:
        super().__init__(name=f"worker-{work_dir.name}", daemon=True)
        self.work_dir = work_dir
        self._repo_name = repo_name
        self._gh = gh
        self._registry = registry
        self._membership = membership if membership is not None else RepoMembership()
        self._wake = threading.Event()
        self._abort_task = AbortHandle()
        self._stop = False
        self.crash_error: str | None = None
        self._provider_lock = threading.Lock()
        # Per-repo issue tree cache (closes #812).  Required — hands the
        # same cache to every Worker iteration so it survives Worker
        # crashes; only a fido restart wipes it (which then
        # re-bootstraps via ``Worker._pick_from_cache``'s lazy
        # ``find_all_open_issues`` call).
        self._issue_cache = issue_cache
        self._provider_factory = (
            DefaultProviderFactory(session_system_file=_sub_dir() / "persona.md")
            if provider_factory is None
            else provider_factory
        )
        self._provider: Provider | None
        if provider is not None:
            self._provider = provider
            if session is not None:
                self._provider.agent.attach_session(session)
        elif repo_cfg is not None:
            self._provider = self._provider_factory.create_provider(
                repo_cfg,
                work_dir=work_dir,
                repo_name=repo_name,
                session=session,
            )
        else:
            self._provider = None
        self._session_issue: int | None = session_issue
        self._config = config
        self._repo_cfg = repo_cfg
        self.__dict__["_bootstrap_session"] = session
        # True until the first ``Worker.run()`` returns — flipped after that so
        # the one-shot startup backfill (fix #794) only fires once per thread.
        self._is_first_iteration = True

    @property
    def _session(self) -> PromptSession | None:
        with self._provider_lock:
            provider = self._provider
        if provider is not None:
            return provider.agent.session
        return self.__dict__.get("_bootstrap_session")

    @_session.setter
    def _session(self, session: PromptSession | None) -> None:
        with self._provider_lock:
            provider = self._provider
            if provider is None:
                if self._repo_cfg is None:
                    self.__dict__["_bootstrap_session"] = session
                    return
                provider = self._provider_factory.create_provider(
                    self._repo_cfg,
                    work_dir=self.work_dir,
                    repo_name=self._repo_name,
                    session=session,
                )
                self._provider = provider
        provider.agent.attach_session(session)

    def detach_provider(self) -> Provider | None:
        """Detach and return the owned provider for crash rescue."""
        with self._provider_lock:
            provider = self._provider
            self._provider = None
            return provider

    def current_provider(self) -> Provider | None:
        """Return the currently attached provider, if any."""
        with self._provider_lock:
            return self._provider

    def recover_provider(self) -> bool:
        """Recover the attached provider session, if any."""
        with self._provider_lock:
            provider = self._provider
        if provider is None:
            return False
        return provider.agent.recover_session()

    @property
    def session_owner(self) -> str | None:
        """Name of the thread currently holding the provider session lock, or ``None``.

        Delegates to the active session's ``owner`` field. Returns ``None`` when
        no session exists or the lock is free. Safe to call from any thread —
        reads a volatile field for display only.
        """
        with self._provider_lock:
            provider = self._provider
        return provider.agent.session_owner if provider is not None else None

    @property
    def session_alive(self) -> bool:
        """True if the persistent provider session subprocess is alive.

        Distinct from :attr:`session_owner` — a session that nobody currently
        holds still reports alive so status display can distinguish
        "session exists, idle" from "no session".  Returns ``False`` when no
        session object exists or its subprocess has exited.
        """
        with self._provider_lock:
            provider = self._provider
        return provider.agent.session_alive if provider is not None else False

    @property
    def session_pid(self) -> int | None:
        """PID of the persistent provider session subprocess, or ``None``.

        Reads directly from the tracked session rather than relying on
        pgrep — the tracked session uses ``sub/persona.md`` (outside
        ``fido_dir``) as its system prompt, which
        the pgrep-based status heuristic can't find.
        """
        with self._provider_lock:
            provider = self._provider
        return provider.agent.session_pid if provider is not None else None

    @property
    def session_dropped_count(self) -> int:
        """Number of stale provider session ids dropped by the live session."""
        with self._provider_lock:
            provider = self._provider
        return provider.agent.session_dropped_count if provider is not None else 0

    @property
    def session_sent_count(self) -> int:
        """Number of messages sent to the current session subprocess since spawn."""
        with self._provider_lock:
            provider = self._provider
        return provider.agent.session_sent_count if provider is not None else 0

    @property
    def session_received_count(self) -> int:
        """Number of stream-json events received from the current session subprocess since spawn."""
        with self._provider_lock:
            provider = self._provider
        return provider.agent.session_received_count if provider is not None else 0

    def wake(self) -> None:
        """Signal the thread to wake up and check for work immediately."""
        self._wake.set()

    def abort_task(self, task_id: str | None = None) -> None:
        """Signal the worker to abort *task_id* after provider_run returns.

        ``task_id=None`` is the legacy untargeted form, used by external
        entry points that want to abort whichever task is running.  Real
        callers (preempt, rescope) always know the task they're aborting
        and should pass it so a leaked abort cannot clobber a different,
        unrelated task on the next loop iteration.
        """
        self._abort_task.request(task_id)
        self._wake.set()

    def stop(self) -> None:
        """Request the thread to exit after the current iteration."""
        self._stop = True
        self._wake.set()

    def _ensure_provider(self) -> Provider:
        """Return the owned provider, creating the configured provider if needed."""
        with self._provider_lock:
            provider = self._provider
            if provider is None:
                if self._repo_cfg is None:
                    raise RuntimeError(
                        "worker thread provider requires explicit repo_cfg"
                    )
                provider = self._provider_factory.create_provider(
                    self._repo_cfg,
                    work_dir=self.work_dir,
                    repo_name=self._repo_name,
                    session=self._session,
                )
                self._provider = provider
            return provider

    def _create_session(self) -> PromptSession:
        """Eagerly create the persistent provider session for this thread.

        Seeds the provider with the durable session id persisted in
        ``state.json`` so a fido self-restart can resume the same
        provider conversation instead of starting fresh (fix for #649).
        When the persisted id is no longer recognised by the provider
        (e.g. evicted server-side), the provider's own stale-session
        recovery path transparently creates a new one.
        """
        provider = self._ensure_provider()
        persisted_session_id = self._load_persisted_session_id()
        provider.agent.ensure_session(
            provider.agent.voice_model, session_id=persisted_session_id
        )
        session = provider.agent.session
        if session is None:
            raise RuntimeError("provider.ensure_session() returned no session")
        return session

    def _resolve_fido_dir(self) -> Path | None:
        """Return the ``.git/fido`` directory for this worker's work_dir,
        or ``None`` when *work_dir* is not a git worktree (e.g. pytest
        tmp_path fixtures that don't initialise a repo).  Callers that
        want to read or write ``state.json`` should treat ``None`` as
        "no persistence available" rather than raising.
        """
        try:
            return _resolve_git_dir(self.work_dir) / "fido"
        except subprocess.CalledProcessError, FileNotFoundError, OSError:
            return None

    def _load_persisted_session_id(self) -> str | None:
        fido_dir = self._resolve_fido_dir()
        if fido_dir is None:
            return None
        try:
            data = State(fido_dir).load()
        except OSError:
            return None
        sid = data.get("session_id")
        return sid if isinstance(sid, str) and sid else None

    def _persist_session_id(self) -> None:
        """Write the live session's durable id back to ``state.json`` so the
        next fido run can resume the same provider conversation.  Silently
        no-ops when no live session / no session id / no persistence target.
        """
        with self._provider_lock:
            provider = self._provider
        if provider is None:
            return
        session = provider.agent.session
        if session is None:
            return
        sid = getattr(session, "session_id", None)
        if not isinstance(sid, str) or not sid:
            return
        fido_dir = self._resolve_fido_dir()
        if fido_dir is None:
            return
        try:
            with State(fido_dir).modify() as data:
                if data.get("session_id") != sid:
                    data["session_id"] = sid
        except OSError as exc:
            log.warning("failed to persist session_id: %s", exc)

    def _retire_poisoned_session(self) -> None:
        """Discard the persisted and live session after a context-window overflow.

        The session thread is full and cannot be used again — re-using the same
        ``session_id`` would produce an immediate ``contextWindowExceeded`` crash.
        This method:
        - removes ``session_id`` from ``state.json`` (so a fido restart doesn't
          reload the poisoned id),
        - calls ``reset()`` on the live session (so the next iteration starts a
          fresh provider thread),
        - clears ``_session_issue`` (so the next iteration uses
          ``TurnSessionMode.FRESH`` and fully re-establishes the system prompt).
        """
        log.warning(
            "context overflow — retiring poisoned session for %s", self._repo_name
        )
        fido_dir = self._resolve_fido_dir()
        if fido_dir is not None:
            try:
                with State(fido_dir).modify() as data:
                    data.pop("session_id", None)
            except OSError as exc:
                log.warning(
                    "failed to clear session_id after context overflow: %s", exc
                )
        with self._provider_lock:
            provider = self._provider
        if provider is not None:
            session = provider.agent.session
            if session is not None:
                session.reset()
        self._session_issue = None

    def run(self) -> None:
        """Main loop — runs until :meth:`stop` is called."""
        _thread_repo.repo_name = self._repo_name.split("/")[-1]
        set_thread_repo(self._repo_name)
        set_thread_kind("worker")
        try:
            while not self._stop:
                if self._registry is not None:
                    self._registry.report_activity(
                        self._repo_name, "scanning for work", busy=False
                    )
                provider = self._ensure_provider()
                session = provider.agent.session
                if session is None:
                    session = self._create_session()
                    if provider.agent.session is not session:
                        provider.agent.attach_session(session)
                worker = Worker(
                    self.work_dir,
                    self._gh,
                    self._abort_task,
                    self._repo_name,
                    self._registry,
                    self._membership,
                    session=session,
                    session_issue=self._session_issue,
                    config=self._config,
                    repo_cfg=self._repo_cfg,
                    provider_factory=self._provider_factory,
                    first_iteration=self._is_first_iteration,
                    issue_cache=self._issue_cache,
                )
                worker._provider = provider  # pyright: ignore[reportPrivateUsage]
                worker._provider_agent = provider.agent  # pyright: ignore[reportPrivateUsage]
                try:
                    try:
                        result = worker.run()
                    finally:
                        with self._provider_lock:
                            self._provider = worker._provider  # pyright: ignore[reportPrivateUsage]
                        self._session_issue = worker._session_issue  # pyright: ignore[reportPrivateUsage]
                        self._persist_session_id()
                        self._is_first_iteration = False

                    if result == 1:
                        # Did work — loop immediately without waiting.
                        continue

                    if self._registry is not None:
                        waiting_what = (
                            "waiting: lock held"
                            if result == 2
                            else "waiting: no issues found"
                        )
                        self._registry.report_activity(
                            self._repo_name, waiting_what, busy=False
                        )
                    timeout = _RETRY_TIMEOUT if result == 2 else _IDLE_TIMEOUT
                    self._wake.wait(timeout=timeout)
                    self._wake.clear()
                except ContextOverflowError:
                    # The provider session hit its context-window limit.
                    # Retire the poisoned session and immediately retry —
                    # the task stays in_progress and the fresh session picks
                    # it right back up.
                    self._retire_poisoned_session()
        except SessionLeakError:
            # A worker and webhook tried to talk to the same repo's claude
            # at the same time — halt fido so the supervisor restarts fresh
            # rather than let sub-claudes multiply silently.
            log.exception(
                "claude leak detected in worker thread for %s — halting",
                self._repo_name,
            )
            os._exit(3)
        except Exception as exc:
            self.crash_error = f"{type(exc).__name__}: {exc}"
            log.exception("WorkerThread %s: unexpected error", self.name)
            raise
        finally:
            set_thread_kind(None)
            set_thread_repo(None)
            # Only stop the session on orderly shutdown — a crashed thread
            # leaves it alive so the registry can hand it to the replacement.
            if self._stop:
                with self._provider_lock:
                    provider = self._provider
                if provider is not None:
                    provider.agent.stop_session()
