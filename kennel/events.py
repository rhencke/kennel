from __future__ import annotations

import fcntl
import logging
import re
import subprocess
import threading
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


def _is_allowed(user: str, repo_cfg: RepoConfig, config: Config) -> bool:
    """Check if user is a repo collaborator or an allowed bot.

    ``repo_cfg.membership.collaborators`` is populated at server startup
    (``server.populate_memberships``) and excludes the bot itself.
    """
    return user in repo_cfg.membership.collaborators or user in config.allowed_bots


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
        if not _is_allowed(user, repo_cfg, config):
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
        if not _is_allowed(user, repo_cfg, config):
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
                "author": user,
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
        if not _is_allowed(user, repo_cfg, config):
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
                "author": user,
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


def _load_persona(config: Config) -> str:
    """Read persona.md from sub_dir; return empty string if missing."""
    try:
        return (config.sub_dir / "persona.md").read_text()
    except FileNotFoundError:
        return ""


def _open_defer_issue(
    gh: Any, repo: str, pr_url: str, title: str, comment: str
) -> str | None:
    """Create a tracking issue for a DEFER triage result.

    Returns the new issue URL, or None if creation failed.
    """
    issue_body = f"Deferred from {pr_url}\n\n> {comment}" if pr_url else comment
    try:
        url = gh.create_issue(repo, title, issue_body)
        log.info("opened tracking issue for DEFER: %s", url)
        return url
    except Exception:
        log.exception("failed to open tracking issue for DEFER")
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
    prompts = Prompts(_load_persona(config))
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
) -> tuple[bool, str, list[str]]:
    """Triage a comment via Opus, generate a reply via Opus, post it.

    Returns (posted, triage_category, task_titles).
    posted is True only when the reply was successfully sent to GitHub.
    task_titles is a list: one entry for non-task categories (used as reply
    context), or one or more entries for ACT/DO (each becomes a task).
    Uses a per-comment lockfile to prevent races with work.sh.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    info = action.reply_to
    if not info or not action.comment_body:
        return (False, "ACT", [action.comment_body or action.prompt])

    # Per-comment lock — prevents kennel and work.sh from both replying
    cid = info.get("comment_id")
    if cid:
        lock_path = _comment_lock(repo_cfg.work_dir, cid)
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.info("comment %s locked by another process — skipping", cid)
            lock_fd.close()
            return (False, "ACT", [action.comment_body[:80]])
    else:
        lock_fd = None

    gh = _gh if _gh is not None else get_github()
    prompts = Prompts(_load_persona(config))
    comment = action.comment_body

    context: dict[str, Any] = dict(action.context) if action.context else {}

    # Always fetch the full thread for this comment
    if info.get("repo") and info.get("pr") and info.get("comment_id"):
        thread = gh.fetch_comment_thread(info["repo"], info["pr"], info["comment_id"])
        if thread:
            context["comment_thread"] = thread
            log.info("fetched %d comment(s) in thread for context", len(thread))

    # Enrich context with sibling threads when the comment needs more context
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
    category, titles = _triage(
        comment, action.is_bot, context, _print_prompt=_print_prompt
    )
    log.info("triage: %s — %s", category, titles)

    # Step 2: For DEFER, open a tracking issue before crafting the reply
    issue_url: str | None = None
    if category == "DEFER" and info.get("repo"):
        pr_url = f"https://github.com/{info['repo']}/pull/{info['pr']}"
        issue_url = _open_defer_issue(gh, info["repo"], pr_url, titles[0], comment)

    # Step 3: Opus reply based on triage
    instr = reply_instruction(
        category, comment, ", ".join(titles), context, issue_url=issue_url
    )

    log.info(
        "generating %s reply for PR #%s comment %s",
        category,
        info["pr"],
        info["comment_id"],
    )
    body = _print_prompt(
        prompts.persona_wrap(instr),
        "claude-opus-4-6",
        system_prompt=prompts.reply_system_prompt(),
        timeout=30,
    )

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

    return (posted, category, titles)


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


_MAX_TITLE_LEN = 80


def _summarize_as_action_item(comment_body: str, *, _print_prompt=None) -> str:
    """Ask Opus to convert a comment into a short imperative action-item title.

    If the result is too long, asks Claude to shorten it up to 3 times before
    falling back to hard truncation.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    prompt = (
        "Convert this PR review comment into a short, imperative task title starting with a verb. "
        "Reply with ONLY the title — no category prefix, no punctuation at the end.\n\n"
        f"Comment: {comment_body}"
    )
    result = _print_prompt(prompt, "claude-opus-4-6", timeout=15).strip()
    for _ in range(3):
        if not result or len(result) <= _MAX_TITLE_LEN:
            break
        result = _print_prompt(
            f"Shorten this task title to under {_MAX_TITLE_LEN} characters while keeping it imperative. "
            f"Reply with ONLY the shortened title.\n\nTitle: {result}",
            "claude-opus-4-6",
            timeout=15,
        ).strip()
    return result[:_MAX_TITLE_LEN] if result else comment_body[:_MAX_TITLE_LEN]


def _triage(
    comment_body: str,
    is_bot: bool,
    context: dict[str, Any] | None = None,
    *,
    _print_prompt=None,
) -> tuple[str, list[str]]:
    """Ask Opus to triage a comment. Returns (category, titles).

    A comment may produce zero or many tasks: titles is a list with one entry
    for ANSWER/ASK/DEFER/DUMP (used as reply context), or one or more entries
    for ACT/DO (each becomes a separate work-queue task).
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    prompt = triage_prompt(comment_body, is_bot, context)
    text = _print_prompt(prompt, "claude-opus-4-6", timeout=15)
    category: str | None = None
    titles: list[str] = []
    for line in text.splitlines() if text else []:
        if ":" not in line:
            continue
        prefix, title = line.split(":", 1)
        prefix = prefix.strip().upper()
        title = title.strip()
        if prefix not in ("ACT", "ASK", "ANSWER", "DO", "DEFER", "DUMP"):
            continue
        if category is None:
            category = prefix
        if prefix == category and title:
            titles.append(title)
    if category is not None and titles:
        return category, titles
    # Fallback: ACT for humans, DO for bots; summarize comment into action item
    category = "DO" if is_bot else "ACT"
    title = _summarize_as_action_item(comment_body, _print_prompt=_print_prompt)
    return category, [title]


def reply_to_issue_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    *,
    _print_prompt=None,
    _gh=None,
) -> tuple[str, list[str]]:
    """Triage and reply to a top-level PR comment (issue_comment event)."""
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    comment = action.comment_body or ""

    # Extract PR number from prompt
    m = re.search(r"#(\d+)", action.prompt)
    number = m.group(1) if m else ""

    gh = _gh if _gh is not None else get_github()
    repo_full = gh.get_repo_info(cwd=repo_cfg.work_dir)

    # Fetch full conversation history for context
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

    prompts = Prompts(_load_persona(config))
    category, titles = _triage(
        comment, action.is_bot, context or None, _print_prompt=_print_prompt
    )
    log.info("issue comment triage: %s — %s", category, titles)

    # For DEFER, open a tracking issue before crafting the reply
    issue_url: str | None = None
    if category == "DEFER":
        pr_url = f"https://github.com/{repo_full}/pull/{number}" if number else ""
        issue_url = _open_defer_issue(gh, repo_full, pr_url, titles[0], comment)

    instr = issue_reply_instruction(
        category, comment, ", ".join(titles), action.context, issue_url=issue_url
    )

    log.info("generating %s reply for issue comment on PR #%s", category, number)
    body = _print_prompt(
        prompts.persona_wrap(instr),
        "claude-opus-4-6",
        system_prompt=prompts.reply_system_prompt(),
        timeout=30,
    )
    if not body:
        body = "On it!" if category in ("ACT", "DO") else "Noted."

    log.info("posting issue comment reply on PR #%s: %s", number, body[:80])
    try:
        gh.comment_issue(repo_full, number, body)
        log.info("reply posted")
    except Exception:
        log.exception("failed to post issue comment reply")

    # Get comment_id from the dispatch payload (stored in context)
    _cid = (action.context or {}).get("comment_id")
    if _cid:
        maybe_react(
            comment,
            _cid,
            "issues",
            repo_full,
            config,
            _print_prompt=_print_prompt,
            _gh=gh,
        )

    return (category, titles)


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
    from kennel.state import load_state
    from kennel.tasks import list_tasks

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


def _get_commit_summary(work_dir: Path) -> str:
    """Return a short ``git log --oneline`` summary of recent commits.

    Used to give Opus context about what has already been implemented when
    it reorders the pending task list.  Returns an empty string on any error
    (e.g. not a git repository, git not found, timeout).
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _notify_thread_change(
    change: dict[str, Any],
    config: Config,
    *,
    _print_prompt=None,
    _gh=None,
) -> None:
    """Post a brief comment notifying a commenter that their task was rescoped.

    Called for each thread task that was dropped or modified during dependency
    analysis.  Uses Opus (in Fido's voice) to generate the message; falls back
    to a plain factual note if Opus returns nothing.

    Posts via the issue comments API so the notification appears regardless of
    whether the original comment was an inline review comment or a top-level
    PR comment.
    """
    if _print_prompt is None:
        _print_prompt = claude.print_prompt
    gh = _gh if _gh is not None else get_github()

    task = change["task"]
    thread = task.get("thread") or {}
    comment_id = thread.get("comment_id")
    repo = thread.get("repo", "")
    pr = thread.get("pr")
    url = thread.get("url", "")
    author = thread.get("author", "")
    if not (comment_id and repo and pr):
        return

    kind = change["kind"]
    original_title = task.get("title", "")
    prompts = Prompts(_load_persona(config))
    mention = f"@{author} " if author else ""

    if kind == "completed":
        instruction = (
            f"A task originating from a PR comment has been marked done — it was "
            f"covered by work already committed and is no longer in the active queue.\n\n"
            f"Original task: {original_title}\n"
            f"Comment author: {author or '(unknown)'}\n"
            f"Comment: {url}\n\n"
            "Write a very brief PR comment notifying the comment author (mention them "
            "with @username if known) that their task has been marked done because it "
            "was covered by recent commits. Reference the comment URL."
        )
        fallback = (
            f"{mention}FYI — the task from your comment ('{original_title}') has been "
            f"marked done: it was covered by recent commits."
        )
    else:
        new_title = change.get("new_title", "")
        instruction = (
            f"The task you were planning from a PR comment has been updated to "
            f"reflect new requirements.\n\n"
            f"Original task: {original_title}\n"
            f"Updated task: {new_title}\n"
            f"Comment author: {author or '(unknown)'}\n"
            f"Comment: {url}\n\n"
            "Write a very brief PR comment notifying the comment author (mention them "
            "with @username if known) that their original task has been updated. "
            "Reference the comment URL."
        )
        fallback = (
            f"{mention}FYI — the task from your comment has been updated to: "
            f"'{new_title}' based on new requirements."
        )

    body = _print_prompt(
        prompts.persona_wrap(instruction),
        "claude-opus-4-6",
        system_prompt=prompts.reply_system_prompt(),
        timeout=15,
    )
    if not body:
        body = fallback

    try:
        gh.comment_issue(repo, pr, body)
        log.info("notified thread %s (%s)", comment_id, kind)
    except Exception:
        log.exception("failed to notify thread %s", comment_id)


def _reorder_tasks_background(
    work_dir: Path,
    commit_summary: str,
    config: Config,
    repo_cfg: RepoConfig | None = None,
    registry: WorkerRegistry | None = None,
    *,
    _start=threading.Thread.start,
    _gh=None,
) -> None:
    """Run :func:`~kennel.tasks.reorder_tasks` in a daemon background thread.

    Passes an ``_on_changes`` callback so that any thread tasks dropped or
    modified during rescoping trigger a notification reply to the original
    comment.

    If *repo_cfg* and *registry* are provided, also passes an
    ``_on_inprogress_affected`` callback that aborts the running worker whenever
    the in-progress task is dropped or modified by the rescope, so the worker
    loop restarts on the new next task.
    """
    from kennel.tasks import reorder_tasks

    gh = _gh if _gh is not None else get_github()

    def on_changes(changes: list[dict[str, Any]]) -> None:
        for change in changes:
            _notify_thread_change(change, config, _gh=gh)

    kwargs: dict[str, Any] = {"_on_changes": on_changes}
    if registry is not None and repo_cfg is not None:

        def on_inprogress_affected() -> None:
            log.info(
                "reorder_tasks_background: in-progress task affected — aborting %s",
                repo_cfg.name,
            )
            registry.abort_task(repo_cfg.name)

        kwargs["_on_inprogress_affected"] = on_inprogress_affected

    t = threading.Thread(
        target=reorder_tasks,
        args=(work_dir, commit_summary),
        kwargs=kwargs,
        name=f"reorder-{work_dir.name}",
        daemon=True,
    )
    _start(t)


def create_task(
    prompt: str,
    config: Config,
    repo_cfg: RepoConfig,
    thread: dict[str, Any] | None = None,
    registry: WorkerRegistry | None = None,
    *,
    _get_commit_summary_fn=_get_commit_summary,
    _reorder_background_fn=_reorder_tasks_background,
) -> dict[str, Any]:
    """Write a task to the shared task file, then trigger sync.

    PR comment tasks (those with a thread) are added to the task list at the
    position determined by the rescoping reorder — they receive spec-level
    priority (first in list wins among non-CI tasks).

    When *thread* is set (a PR comment task), also triggers a background
    dependency-analysis reorder via Opus so that remaining spec tasks are
    resequenced to account for the new requirement.  Spec tasks created during
    initial setup are not rescoped — the planner already ordered them.

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
    if thread:
        commit_summary = _get_commit_summary_fn(repo_cfg.work_dir)
        _reorder_background_fn(
            repo_cfg.work_dir, commit_summary, config, repo_cfg, registry
        )
    if registry is not None:
        _maybe_abort_for_new_task(repo_cfg, new_task, registry)
    return new_task


def launch_sync(config: Config, repo_cfg: RepoConfig, *, _gh=None) -> None:
    """Sync tasks.json → PR body in a background thread."""
    from kennel.tasks import sync_tasks_background

    gh = _gh if _gh is not None else get_github()
    sync_tasks_background(repo_cfg.work_dir, gh)
    log.info("sync-tasks launched")


def launch_worker(repo_cfg: RepoConfig, registry: WorkerRegistry) -> None:
    """Wake the per-repo WorkerThread via the registry."""
    log.info("waking worker thread for %s", repo_cfg.name)
    registry.wake(repo_cfg.name)
