from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kennel import claude
from kennel.config import Config, RepoConfig
from kennel.github import get_github
from kennel.prompts import (
    Prompts,
    issue_reply_instruction,
    reply_instruction,
    triage_prompt,
)
from kennel.registry import WorkerRegistry
from kennel.tasks import add_task
from kennel.types import TaskType

log = logging.getLogger(__name__)


@dataclass
class Action:
    prompt: str
    reply_to: dict[str, Any] | None = None  # {repo, pr, comment_id}
    review_comments: dict[str, Any] | None = None  # {repo, pr, review_id}
    comment_body: str | None = None
    is_bot: bool = False
    context: dict[str, Any] | None = None  # {pr_title, file, diff_hunk, line, pr_body}
    thread: dict[str, Any] | None = (
        None  # {repo, pr, comment_id} for task prioritisation
    )


def _is_allowed(user: str, payload: dict[str, Any], config: Config) -> bool:
    """Check if user is the repo owner or an allowed bot."""
    owner = payload.get("repository", {}).get("owner", {}).get("login", "")
    return user == owner or user in config.allowed_bots


def dispatch(
    event: str, payload: dict[str, Any], config: Config, repo_cfg: RepoConfig
) -> Action | None:
    """Map a GitHub webhook event to an action. Returns None if ignored."""
    action = payload.get("action", "")
    repo = payload.get("repository", {}).get("full_name", "")

    if event == "ping":
        log.info("ping received — hook_id=%s", payload.get("hook_id"))
        return None

    if event == "issues" and action == "assigned":
        assignee = payload.get("assignee", {}).get("login", "")
        issue = payload.get("issue", {})
        number = issue.get("number")
        title = issue.get("title", "")
        if not number:
            return None
        log.info("issue #%s assigned to %s: %s", number, assignee, title)
        return Action(prompt=f"New issue #{number} assigned to {assignee}: {title}")

    if event == "pull_request_review" and action == "submitted":
        review = payload.get("review", {})
        pr = payload.get("pull_request", {})
        number = pr.get("number")
        state = review.get("state", "")
        user = review.get("user", {}).get("login", "")
        review_id = review.get("id")
        if not number:
            return None
        if not _is_allowed(user, payload, config):
            log.debug("ignoring review on PR #%s by %s (not allowed)", number, user)
            return None
        log.info("review on PR #%s: %s by %s", number, state, user)
        return Action(
            prompt=f"Review on PR #{number}: {state} by {user}",
            review_comments={"repo": repo, "pr": number, "review_id": review_id}
            if review_id
            else None,
        )

    if event == "pull_request_review_comment" and action == "created":
        comment = payload.get("comment", {})
        pr = payload.get("pull_request", {})
        number = pr.get("number")
        user = comment.get("user", {}).get("login", "")
        comment_id = comment.get("id")
        if user.lower() in ("fidocancode", "fido-can-code"):
            log.debug("ignoring own comment on PR #%s", number)
            return None
        if not number:
            return None
        if not _is_allowed(user, payload, config):
            log.debug("ignoring comment on PR #%s by %s (not allowed)", number, user)
            return None
        comment_body = comment.get("body", "") or ""
        log.info("comment on PR #%s by %s: %s", number, user, comment_body[:80])
        is_bot = user.endswith("[bot]")
        return Action(
            prompt=f"Review comment on PR #{number} by {user} ({'bot' if is_bot else 'human/owner'}):\n\n{comment_body}",
            reply_to={
                "repo": repo,
                "pr": number,
                "comment_id": comment_id,
                "url": comment.get("html_url", ""),
            },
            comment_body=comment_body,
            is_bot=is_bot,
            context={
                "pr_title": pr.get("title", ""),
                "pr_body": pr.get("body", "") or "",
                "file": comment.get("path", ""),
                "line": comment.get("line"),
                "diff_hunk": comment.get("diff_hunk", ""),
            },
        )

    if event == "issue_comment" and action == "created":
        comment = payload.get("comment", {})
        issue = payload.get("issue", {})
        user = comment.get("user", {}).get("login", "")
        pr = issue.get("pull_request")
        if not pr:
            log.debug("issue_comment on non-PR issue — ignoring")
            return None
        if user.lower() in ("fidocancode", "fido-can-code"):
            log.debug("ignoring own comment on PR")
            return None
        if not _is_allowed(user, payload, config):
            log.debug("ignoring comment by %s (not allowed)", user)
            return None
        number = issue.get("number")
        comment_body = comment.get("body", "") or ""
        comment_id = comment.get("id")
        is_bot = user.endswith("[bot]")
        log.info("PR comment on #%s by %s: %s", number, user, comment_body[:80])
        return Action(
            prompt=f"PR top-level comment on #{number} by {user}:\n\n{comment_body}",
            reply_to=None,  # top-level comments use issues API, not pulls
            comment_body=comment_body,
            is_bot=is_bot,
            context={
                "pr_title": issue.get("title", ""),
                "pr_body": issue.get("body", "") or "",
                "comment_id": comment_id,
            },
            thread={
                "repo": repo,
                "pr": number,
                "comment_id": comment_id,
                "url": comment.get("html_url", ""),
            }
            if number and comment_id
            else None,
        )

    if event == "check_run" and action == "completed":
        check = payload.get("check_run", {})
        conclusion = check.get("conclusion", "")
        if conclusion not in ("failure", "timed_out"):
            log.debug("check_run completed with %s — ignoring", conclusion)
            return None
        name = check.get("name", "")
        prs = check.get("pull_requests", [])
        pr_nums = [pr.get("number") for pr in prs if pr.get("number")]
        log.info("CI failure: %s (%s) on PRs %s", name, conclusion, pr_nums)
        pr_str = ", ".join(f"#{n}" for n in pr_nums) if pr_nums else "unknown PR"
        return Action(prompt=f"CI failure on {pr_str}: {name} ({conclusion})")

    if event == "pull_request" and action == "closed":
        pr = payload.get("pull_request", {})
        if not pr.get("merged"):
            log.debug("PR #%s closed without merge — ignoring", pr.get("number"))
            return None
        number = pr.get("number")
        log.info("PR #%s merged", number)
        return Action(prompt=f"PR #{number} merged — cleanup")

    log.debug("ignored event: %s (action=%s)", event, action)
    return None


def _comment_lock(work_dir: Path, comment_id: int) -> Path:
    """Return path to a per-comment lockfile."""
    lock_dir = work_dir / ".git" / "fido" / "comments"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{comment_id}.lock"


def maybe_react(
    comment_body: str,
    comment_id: int | str,
    comment_type: str,
    repo: str,
    config: Config,
    *,
    _print_prompt=None,
    _gh=None,
) -> None:
    """Let Fido decide whether to react to a comment with an emoji.

    comment_type: 'pulls' for review comments, 'issues' for issue comments.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    persona_path = config.sub_dir / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    prompts = Prompts(persona)
    reaction = (
        _print_prompt(prompts.react_prompt(comment_body), "claude-opus-4-6", timeout=15)
        .lower()
        .split("\n")[0]
        .strip()
    )

    valid = {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
    if reaction not in valid:
        log.debug("fido chose not to react (got: %s)", reaction)
        return

    gh = _gh if _gh is not None else get_github()
    log.info("fido reacts with %s to comment %s", reaction, comment_id)
    try:
        gh.add_reaction(repo, comment_type, comment_id, reaction)
    except Exception:
        log.exception("failed to post reaction")


def reply_to_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    *,
    _print_prompt=None,
    _gh=None,
) -> tuple[bool, str, str]:
    """Triage a comment via Opus, generate a reply via Opus, post it.

    Returns (posted, triage_category, task_title).
    posted is True only when the reply was successfully sent to GitHub.
    Uses a per-comment lockfile to prevent races with work.sh.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    info = action.reply_to
    if not info or not action.comment_body:
        return (False, "ACT", action.comment_body or action.prompt)

    # Per-comment lock — prevents kennel and work.sh from both replying
    import fcntl

    cid = info.get("comment_id")
    if cid:
        lock_path = _comment_lock(repo_cfg.work_dir, cid)
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.info("comment %s locked by another process — skipping", cid)
            lock_fd.close()
            return (False, "ACT", action.comment_body[:80])
    else:
        lock_fd = None

    gh = _gh if _gh is not None else get_github()

    persona_path = config.sub_dir / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    prompts = Prompts(persona)
    comment = action.comment_body

    # Enrich context with sibling threads when the comment needs more context
    context: dict[str, Any] = dict(action.context) if action.context else {}
    if (
        needs_more_context(comment, _print_prompt=_print_prompt)
        and info.get("repo")
        and info.get("pr")
    ):
        siblings = gh.fetch_sibling_threads(info["repo"], info["pr"])
        if siblings:
            context["sibling_threads"] = siblings
            log.info(
                "needs-more-context comment — fetched %d sibling thread(s) for context",
                len(siblings),
            )

    # Step 1: Haiku triage
    category, title = _triage(
        comment, action.is_bot, context, _print_prompt=_print_prompt
    )
    log.info("triage: %s — %s", category, title)

    # Step 2: For DEFER, open a tracking issue before crafting the reply
    issue_url: str | None = None
    if category == "DEFER" and info.get("repo"):
        pr_url = f"https://github.com/{info['repo']}/pull/{info['pr']}"
        issue_body = f"Deferred from {pr_url}\n\n> {comment}"
        try:
            issue_url = gh.create_issue(info["repo"], title, issue_body)
            log.info("opened tracking issue for DEFER: %s", issue_url)
        except Exception:
            log.exception("failed to open tracking issue for DEFER")

    # Step 3: Opus reply based on triage
    instr = reply_instruction(category, comment, title, context, issue_url=issue_url)

    log.info(
        "generating %s reply for PR #%s comment %s",
        category,
        info["pr"],
        info["comment_id"],
    )
    body = _print_prompt(prompts.persona_wrap(instr), "claude-opus-4-6", timeout=30)

    if not body:
        body = (
            "Looking into this now."
            if category in ("ACT", "DO")
            else "Noted — checking on this."
        )

    log.info("posting reply to PR #%s: %s", info["pr"], body[:80])
    posted = False
    try:
        gh.reply_to_review_comment(info["repo"], info["pr"], body, info["comment_id"])
        posted = True
        log.info("reply posted")
    except Exception:
        log.exception("failed to post reply")

    # Maybe react
    maybe_react(
        comment,
        info["comment_id"],
        "pulls",
        info.get("repo", ""),
        config,
        _print_prompt=_print_prompt,
        _gh=gh,
    )

    # For DUMP: also resolve the thread
    if category == "DUMP" and info.get("comment_id"):
        _try_resolve_thread(info, config)

    # Release comment lock (keep file so work.sh sees it was claimed)
    if lock_fd:
        lock_fd.close()

    return (posted, category, title)


def _try_resolve_thread(info: dict[str, Any], config: Config) -> None:
    """Best-effort resolve a review thread via GraphQL."""
    # We'd need the thread node_id — skip for now, work.sh will handle it
    pass


def reply_to_review(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    already_replied: set[int] | None = None,
    *,
    _print_prompt=None,
    _gh=None,
) -> None:
    """Fetch inline comments from a review and reply to each."""
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    info = action.review_comments
    if not info:
        return

    gh = _gh if _gh is not None else get_github()
    log.info(
        "fetching review comments for PR #%s review %s", info["pr"], info["review_id"]
    )
    try:
        comments = gh.get_review_comments(info["repo"], info["pr"], info["review_id"])
    except Exception:
        log.exception("failed to fetch review comments")
        return

    if not comments:
        log.info("no inline comments in review")
        return

    skipped = [
        cid for cid, _body in comments if already_replied and cid in already_replied
    ]
    todo = [
        (cid, body)
        for cid, body in comments
        if not already_replied or cid not in already_replied
    ]
    if skipped:
        log.info("skipping %d already-replied comments", len(skipped))
    if not todo:
        return
    log.info("replying to %d review comments", len(todo))
    for cid, body in todo:
        posted, *_ = reply_to_comment(
            Action(
                prompt=action.prompt,
                reply_to={
                    "repo": info["repo"],
                    "pr": info["pr"],
                    "comment_id": cid,
                },
                comment_body=body,
            ),
            config,
            repo_cfg,
            _print_prompt=_print_prompt,
            _gh=gh,
        )
        if posted and already_replied is not None:
            already_replied.add(cid)


def needs_more_context(comment_body: str, *, _print_prompt=None) -> bool:
    """Ask Haiku whether this comment needs sibling thread context to act on.

    Returns True if Haiku thinks the comment is too vague or cross-referential
    to act on alone (e.g. "same", "ditto", "^"), False otherwise.
    Falls back to False on any error.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    prompt = (
        "A reviewer left this comment on a pull request:\n\n"
        f"{comment_body!r}\n\n"
        "Does this comment need context from sibling review threads to be understood "
        "(e.g. it says 'same', 'ditto', '^', 'here too', or is otherwise too vague "
        "to act on alone)?\n\n"
        "Reply with exactly YES or NO."
    )
    answer = _print_prompt(prompt, "claude-haiku-4-5", timeout=10).upper()
    return answer.startswith("YES")


def _summarize_as_action_item(comment_body: str, *, _print_prompt=None) -> str:
    """Ask Opus to convert a comment into a short imperative action-item title."""
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    prompt = (
        "Convert this PR review comment into a short, imperative task title starting with a verb. "
        "Reply with ONLY the title — no category prefix, no punctuation at the end.\n\n"
        f"Comment: {comment_body}"
    )
    result = _print_prompt(prompt, "claude-opus-4-6", timeout=15).strip()
    return result or comment_body[:80]


def _triage(
    comment_body: str,
    is_bot: bool,
    context: dict[str, Any] | None = None,
    *,
    _print_prompt=None,
) -> tuple[str, str]:
    """Ask Opus to triage a comment. Returns (prefix, title)."""
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    prompt = triage_prompt(comment_body, is_bot, context)
    text = _print_prompt(prompt, "claude-opus-4-6", timeout=15)
    line = text.splitlines()[0] if text else ""
    if ":" in line:
        prefix, title = line.split(":", 1)
        prefix = prefix.strip().upper()
        title = title.strip()
        if prefix in ("ACT", "ASK", "ANSWER", "DO", "DEFER", "DUMP"):
            return prefix, title
    # Fallback: ACT for humans, DO for bots; summarize comment into action item
    category = "DO" if is_bot else "ACT"
    title = _summarize_as_action_item(comment_body, _print_prompt=_print_prompt)
    return category, title


def reply_to_issue_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    *,
    _print_prompt=None,
    _gh=None,
) -> tuple[str, str]:
    """Triage and reply to a top-level PR comment (issue_comment event)."""
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    comment = action.comment_body or ""

    # Extract PR number from prompt
    import re

    m = re.search(r"#(\d+)", action.prompt)
    number = m.group(1) if m else ""

    # Fetch full conversation history for context
    gh = _gh if _gh is not None else get_github()
    conversation_context = ""
    if number:
        try:
            all_comments = gh.get_issue_comments(repo_cfg.name, int(number))
            preceding = [c for c in all_comments if c.get("body", "") != comment]
            if preceding:
                lines = [
                    f"{c.get('user', {}).get('login', '?')}: {c.get('body', '')}"
                    for c in preceding
                ]
                conversation_context = (
                    "\n\nFull conversation on this issue/PR:\n" + "\n".join(lines)
                )
        except Exception:
            pass  # best-effort

    # Merge conversation context into triage context
    context = dict(action.context) if action.context else {}
    if conversation_context:
        context["conversation"] = conversation_context

    persona_path = config.sub_dir / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    prompts = Prompts(persona)
    category, title = _triage(
        comment, action.is_bot, context or None, _print_prompt=_print_prompt
    )
    log.info("issue comment triage: %s — %s", category, title)

    gh = _gh if _gh is not None else get_github()

    # For DEFER, open a tracking issue before crafting the reply
    issue_url: str | None = None
    if category == "DEFER":
        repo_full = gh.get_repo_info(cwd=repo_cfg.work_dir)
        pr_url = f"https://github.com/{repo_full}/pull/{number}" if number else ""
        issue_body = f"Deferred from {pr_url}\n\n> {comment}" if pr_url else comment
        try:
            issue_url = gh.create_issue(repo_full, title, issue_body)
            log.info("opened tracking issue for DEFER: %s", issue_url)
        except Exception:
            log.exception("failed to open tracking issue for DEFER")

    instr = issue_reply_instruction(
        category, comment, title, action.context, issue_url=issue_url
    )

    log.info("generating %s reply for issue comment on PR #%s", category, number)
    body = _print_prompt(prompts.persona_wrap(instr), "claude-opus-4-6", timeout=30)
    if not body:
        body = "On it!" if category in ("ACT", "DO") else "Noted."

    log.info("posting issue comment reply on PR #%s: %s", number, body[:80])
    try:
        repo_full = gh.get_repo_info(cwd=repo_cfg.work_dir)
        gh.comment_issue(repo_full, number, body)
        log.info("reply posted")
    except Exception:
        log.exception("failed to post issue comment reply")

    # Get comment_id from the dispatch payload (stored in context)
    _cid = (action.context or {}).get("comment_id")
    if _cid:
        repo_full = gh.get_repo_info(cwd=repo_cfg.work_dir)
        maybe_react(
            comment,
            _cid,
            "issues",
            repo_full,
            config,
            _print_prompt=_print_prompt,
            _gh=gh,
        )

    return (category, title)


_TYPE_PRIORITY = {TaskType.CI: 0, TaskType.THREAD: 1, TaskType.SPEC: 2}


def _maybe_abort_for_new_task(
    repo_cfg: RepoConfig,
    new_task: dict[str, Any],
    registry: WorkerRegistry,
) -> None:
    """Abort the current task if the new task has higher priority.

    Priority is deterministic by type: ci > thread > spec.
    A higher-priority task always preempts — the current task is kept
    pending for later (ABORT_KEEP).  Equal or lower priority does not
    preempt.
    """
    from kennel.tasks import list_tasks
    from kennel.worker import load_state

    fido_dir = repo_cfg.work_dir / ".git" / "fido"
    if not (fido_dir / "state.json").exists():
        return
    state = load_state(fido_dir)
    current_task_id = state.get("current_task_id")
    if not current_task_id:
        return

    task_list = list_tasks(repo_cfg.work_dir)
    current_task = next((t for t in task_list if t["id"] == current_task_id), None)
    if current_task is None:
        return

    new_priority = _TYPE_PRIORITY.get(new_task.get("type", "spec"), 2)
    current_priority = _TYPE_PRIORITY.get(current_task.get("type", "spec"), 2)

    if new_priority < current_priority:
        log.info(
            "preempt: %s task interrupts %s task — aborting %s",
            new_task.get("type", "?"),
            current_task.get("type", "?"),
            current_task.get("title", "")[:60],
        )
        registry.abort_task(repo_cfg.name)


def create_task(
    prompt: str,
    config: Config,
    repo_cfg: RepoConfig,
    thread: dict[str, Any] | None = None,
    registry: WorkerRegistry | None = None,
) -> dict[str, Any]:
    """Write a task to the shared task file, then trigger sync.

    PR comment tasks (those with a thread) carry a thread payload that causes
    ``_pick_next_task`` to prioritise them as NEXT (second only to CI failures),
    without inserting them out-of-order in the list.

    If *registry* is given, checks whether the new task has higher priority
    than the current in-progress task; if so, signals abort so the worker
    picks up the higher-priority task.

    Returns the new task dict.
    """
    task_type = TaskType.THREAD if thread else TaskType.SPEC
    log.info("creating task: %s", prompt[:100])
    new_task = add_task(
        repo_cfg.work_dir, title=prompt, task_type=task_type, thread=thread
    )
    launch_sync(config, repo_cfg)
    if registry is not None:
        _maybe_abort_for_new_task(repo_cfg, new_task, registry)
    return new_task


def launch_sync(config: Config, repo_cfg: RepoConfig, *, _gh=None) -> None:
    """Sync tasks.json → PR body in a background thread."""
    from kennel.worker import sync_tasks_background

    gh = _gh if _gh is not None else get_github()
    sync_tasks_background(repo_cfg.work_dir, gh)
    log.info("sync-tasks launched")


def launch_worker(repo_cfg: RepoConfig, registry: WorkerRegistry) -> None:
    """Wake the per-repo WorkerThread via the registry."""
    log.info("waking worker thread for %s", repo_cfg.name)
    registry.wake(repo_cfg.name)
