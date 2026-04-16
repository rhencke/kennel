from __future__ import annotations

import fcntl
import logging
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kennel import reply_promises
from kennel.claude import ClaudeClient, set_thread_repo
from kennel.config import Config, RepoConfig
from kennel.github import GitHub
from kennel.prompts import NO_TOOLS_CLAUSE, Prompts
from kennel.provider import ProviderAgent
from kennel.registry import WorkerRegistry
from kennel.tasks import Tasks
from kennel.types import TaskType

log = logging.getLogger(__name__)

# Per-work_dir coalescing state for _reorder_tasks_background.
# Ensures at most one Opus call in-flight + one pending per repo.
_reorder_coalesce: dict[str, dict[str, Any]] = {}
_reorder_coalesce_lock = threading.Lock()


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


def _pr_number_from_api_url(url: str, kind: str) -> int:
    """Extract a PR/issue number from a GitHub API URL."""
    pattern = r"/issues/(\d+)$" if kind == "issues" else r"/pulls/(\d+)$"
    match = re.search(pattern, url)
    if match is None:
        raise ValueError(f"invalid GitHub API URL for {kind}: {url!r}")
    return int(match.group(1))


def _build_review_comment_action(
    repo: str,
    pr_number: int,
    pr_title: str,
    pr_body: str,
    comment: dict[str, Any],
    *,
    comment_body: str | None = None,
) -> Action:
    """Rebuild a review-comment Action from live GitHub state."""
    user = comment["user"]["login"]
    body = comment_body if comment_body is not None else (comment["body"] or "")
    is_bot = user.endswith("[bot]")
    return Action(
        prompt=(
            f"Review comment on PR #{pr_number} by {user}"
            f" ({'bot' if is_bot else 'human/owner'}):\n\n{body}"
        ),
        reply_to={
            "repo": repo,
            "pr": pr_number,
            "comment_id": comment["id"],
            "url": comment["html_url"],
            "author": user,
            "comment_type": "pulls",
        },
        comment_body=body,
        is_bot=is_bot,
        context={
            "pr_title": pr_title,
            "pr_body": pr_body,
            "file": comment["path"],
            "line": comment["line"],
            "diff_hunk": comment["diff_hunk"],
        },
    )


def _build_issue_comment_action(
    repo: str,
    pr_number: int,
    pr_title: str,
    pr_body: str,
    comment: dict[str, Any],
) -> Action:
    """Rebuild a top-level PR-comment Action from live GitHub state."""
    user = comment["user"]["login"]
    body = comment["body"] or ""
    is_bot = user.endswith("[bot]")
    comment_id = int(comment["id"])
    return Action(
        prompt=f"PR top-level comment on #{pr_number} by {user}:\n\n{body}",
        reply_to=None,
        comment_body=body,
        is_bot=is_bot,
        context={
            "pr_title": pr_title,
            "pr_body": pr_body,
            "comment_id": comment_id,
        },
        thread={
            "repo": repo,
            "pr": pr_number,
            "comment_id": comment_id,
            "url": comment["html_url"],
            "author": user,
            "comment_type": "issues",
        },
    )


def _apply_reply_result(
    category: str,
    titles: list[str],
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    thread: dict[str, Any] | None,
    registry: WorkerRegistry | None,
) -> None:
    """Apply ACT/DO titles from a recovered reply just like webhook handling."""
    if category in ("DUMP", "ANSWER", "ASK", "DEFER"):
        return
    for title in titles:
        create_task(
            title,
            config,
            repo_cfg,
            gh,
            thread=thread,
            registry=registry,
        )


def recover_reply_promises(
    fido_dir: Path,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    pr_number: int,
    *,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
    registry: WorkerRegistry | None = None,
) -> bool:
    """Recover queued webhook replies for the current PR from promise files."""
    promises = reply_promises.list_reply_promises(fido_dir)
    if not promises:
        return False

    pr_issue = gh.view_issue(repo_cfg.name, pr_number)
    pr_title = pr_issue["title"]
    pr_body = pr_issue["body"] or ""
    processed_any = False
    handled_keys: set[tuple[str, int]] = set()
    promise_by_key = {
        (promise.comment_type, promise.comment_id): promise for promise in promises
    }

    pull_entries: dict[tuple[str, int], tuple[dict[str, Any], int, int]] = {}
    issue_entries: dict[tuple[str, int], tuple[dict[str, Any], int]] = {}

    for promise in promises:
        key = (promise.comment_type, promise.comment_id)
        if promise.comment_type == "pulls":
            comment = gh.get_pull_comment(repo_cfg.name, promise.comment_id)
            if comment is None:
                promise.path.unlink()
                handled_keys.add(key)
                continue
            comment_pr = _pr_number_from_api_url(comment["pull_request_url"], "pulls")
            root_id = (
                int(comment["in_reply_to_id"])
                if "in_reply_to_id" in comment and comment["in_reply_to_id"] is not None
                else int(comment["id"])
            )
            pull_entries[key] = (comment, comment_pr, root_id)
        else:
            comment = gh.get_issue_comment(repo_cfg.name, promise.comment_id)
            if comment is None:
                promise.path.unlink()
                handled_keys.add(key)
                continue
            comment_pr = _pr_number_from_api_url(comment["issue_url"], "issues")
            issue_entries[key] = (comment, comment_pr)

    for promise in promises:
        key = (promise.comment_type, promise.comment_id)
        if key in handled_keys:
            continue

        if promise.comment_type == "issues":
            comment, comment_pr = issue_entries[key]
            if comment_pr != pr_number:
                continue
            action = _build_issue_comment_action(
                repo_cfg.name, pr_number, pr_title, pr_body, comment
            )
            category, titles = reply_to_issue_comment(
                action,
                config,
                repo_cfg,
                gh,
                claude_client=claude_client,
                prompts=prompts,
            )
            reply_promises.remove_reply_promise(
                fido_dir, promise.comment_type, promise.comment_id
            )
            handled_keys.add(key)
            _apply_reply_result(
                category,
                titles,
                config,
                repo_cfg,
                gh,
                thread=action.thread,
                registry=registry,
            )
            processed_any = True
            continue

        comment, comment_pr, root_id = pull_entries[key]
        if comment_pr != pr_number:
            continue

        group: list[tuple[reply_promises.ReplyPromise, dict[str, Any]]] = []
        for candidate_key, (
            candidate_comment,
            candidate_pr,
            candidate_root_id,
        ) in pull_entries.items():
            if candidate_key in handled_keys:
                continue
            if candidate_pr == pr_number and candidate_root_id == root_id:
                group.append((promise_by_key[candidate_key], candidate_comment))

        combined_parts: list[str] = []
        for _, group_comment in group:
            body = group_comment["body"] or ""
            if body and body not in combined_parts:
                combined_parts.append(body)
        combined_body = "\n\n---\n\n".join(combined_parts) if combined_parts else None
        representative = group[-1][1]
        action = _build_review_comment_action(
            repo_cfg.name,
            pr_number,
            pr_title,
            pr_body,
            representative,
            comment_body=combined_body,
        )
        category, titles = reply_to_comment(
            action,
            config,
            repo_cfg,
            gh,
            claude_client=claude_client,
            prompts=prompts,
        )
        for group_promise, _ in group:
            reply_promises.remove_reply_promise(
                fido_dir, group_promise.comment_type, group_promise.comment_id
            )
            handled_keys.add((group_promise.comment_type, group_promise.comment_id))
        _apply_reply_result(
            category,
            titles,
            config,
            repo_cfg,
            gh,
            thread=action.reply_to,
            registry=registry,
        )
        processed_any = True

    return processed_any


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
                "comment_type": "pulls",
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
                "comment_type": "issues",
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


def _open_defer_issue(gh: Any, repo: str, pr_url: str, title: str, comment: str) -> str:
    """Create a tracking issue for a DEFER triage result.

    Returns the new issue URL.  Raises on any creation failure so the caller
    fails closed rather than crafting a reply that references a missing issue.
    """
    issue_body = f"Deferred from {pr_url}\n\n> {comment}" if pr_url else comment
    url = gh.create_issue(repo, title, issue_body)
    log.info("opened tracking issue for DEFER: %s", url)
    return url


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
    gh: GitHub,
    *,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> None:
    """Let Fido decide whether to react to a comment with an emoji.

    comment_type: 'pulls' for review comments, 'issues' for issue comments.
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    if prompts is None:
        prompts = Prompts(_load_persona(config))
    reaction = (
        claude_client.print_prompt(
            prompts.react_prompt(comment_body), "claude-opus-4-6"
        )
        .lower()
        .split("\n")[0]
        .strip()
    )

    valid = {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
    if reaction not in valid:
        log.debug("fido chose not to react (got: %s)", reaction)
        return

    log.info("fido reacts with %s to comment %s", reaction, comment_id)
    try:
        gh.add_reaction(repo, comment_type, comment_id, reaction)
    except Exception:
        log.exception("failed to post reaction")


def reply_to_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Triage a comment via Opus, generate a reply via Opus, post it.

    Returns (triage_category, task_titles).
    task_titles is a list: one entry for non-task categories (used as reply
    context), or one or more entries for ACT/DO (each becomes a task).
    Uses a per-comment lockfile to prevent concurrent replies.
    Raises on reply-post failure so callers fail closed.
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    if prompts is None:
        prompts = Prompts(_load_persona(config))
    info = action.reply_to
    if not info or not action.comment_body:
        return ("ACT", [action.comment_body or action.prompt])

    # Per-comment lock — prevents concurrent replies
    cid = info.get("comment_id")
    if cid:
        lock_path = _comment_lock(repo_cfg.work_dir, cid)
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.info("comment %s locked by another process — skipping", cid)
            lock_fd.close()
            return ("ACT", [action.comment_body[:80]])
    else:
        lock_fd = None

    comment = action.comment_body

    context: dict[str, Any] = dict(action.context) if action.context else {}

    # Always fetch the full thread for this comment.
    # Normalize to list so root_body extraction below is type-safe.
    thread_comments: list[dict[str, Any]] = []
    if info.get("repo") and info.get("pr") and info.get("comment_id"):
        fetched = gh.fetch_comment_thread(info["repo"], info["pr"], info["comment_id"])
        if fetched:
            thread_comments = list(fetched)
            context["comment_thread"] = thread_comments
            log.info(
                "fetched %d comment(s) in thread for context", len(thread_comments)
            )

    # Root comment body — used for task title generation.
    # When the webhook fires on a reply (e.g. "Yes" or "Woof, you're right!"),
    # the task title should describe the reviewer's original feedback, not the reply.
    root_body = thread_comments[0].get("body", comment) if thread_comments else comment

    # Enrich context with sibling threads when the comment needs more context
    if (
        needs_more_context(comment, claude_client=claude_client)
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

    # Step 1: Haiku triage (on the triggering comment to determine category)
    category, titles = _triage(
        comment, action.is_bot, context, claude_client=claude_client, prompts=prompts
    )
    log.info("triage: %s — %s", category, titles)

    # Step 1b: Always derive task titles from the root comment body for action
    # categories.  The originating PR comment is the source of truth for what
    # was requested; triage may have run on a short reply body ("Yes", "Done")
    # that produces a poor title.  Using root_body here ensures the task always
    # reflects what the reviewer originally asked.
    if category in ("ACT", "DO"):
        log.info("deriving task title from root comment")
        titles = [_summarize_as_action_item(root_body, claude_client=claude_client)]

    # Step 2: For DEFER, open a tracking issue before crafting the reply.
    # Raises on failure so we don't craft a reply referencing a missing issue.
    issue_url: str | None = None
    if category == "DEFER" and info.get("repo"):
        pr_url = f"https://github.com/{info['repo']}/pull/{info['pr']}"
        issue_url = _open_defer_issue(gh, info["repo"], pr_url, titles[0], comment)

    # Step 3: Opus reply based on triage
    instr = prompts.reply_instruction(
        category, comment, ", ".join(titles), context, issue_url=issue_url
    )

    log.info(
        "generating %s reply for PR #%s comment %s",
        category,
        info["pr"],
        info["comment_id"],
    )
    body = claude_client.print_prompt(
        prompts.persona_wrap(instr),
        "claude-opus-4-6",
        system_prompt=prompts.reply_system_prompt(),
    )
    if not body:
        raise ValueError(
            f"review-comment reply: print_prompt returned empty for PR #{info['pr']}"
        )

    # Edit the last Fido reply only if it is the most recent comment in the thread
    # (i.e. no human has spoken since). If a human posted a new comment after
    # Fido's last reply, post a fresh reply so the conversation stays coherent.
    _fido_logins = {"fidocancode", "fido-can-code"}
    last_thread_author = (
        thread_comments[-1].get("author", "").lower() if thread_comments else ""
    )
    last_fido_id = next(
        (
            c["id"]
            for c in reversed(thread_comments)
            if c.get("author", "").lower() in _fido_logins
        ),
        None,
    )
    if last_fido_id and last_thread_author in _fido_logins:
        log.info("editing last fido reply %s on PR #%s", last_fido_id, info["pr"])
        gh.edit_review_comment(info["repo"], last_fido_id, body)
        log.info("reply edited")
    else:
        log.info("posting reply to PR #%s: %s", info["pr"], body[:80])
        gh.reply_to_review_comment(info["repo"], info["pr"], body, info["comment_id"])
        log.info("reply posted")

    # Maybe react
    maybe_react(
        comment,
        info["comment_id"],
        "pulls",
        info.get("repo", ""),
        config,
        gh,
        claude_client=claude_client,
        prompts=prompts,
    )

    # For DUMP: also resolve the thread
    if category == "DUMP" and info.get("comment_id"):
        _try_resolve_thread(info, config)

    # Release comment lock (keep file so concurrent callers see it was claimed)
    if lock_fd:
        lock_fd.close()

    return (category, titles)


def _try_resolve_thread(info: dict[str, Any], config: Config) -> None:
    """Best-effort resolve a review thread via GraphQL."""
    # Thread node_id not available in webhook payload — skip
    pass


def reply_to_review(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    already_replied: set[int] | None = None,
    *,
    claude_client: ClaudeClient | None = None,
    prompts: Prompts | None = None,
) -> None:
    """No-op for inline comments — they are handled per-comment.

    GitHub fires both ``pull_request_review`` (this handler) and a separate
    ``pull_request_review_comment`` for each inline comment within the
    review.  Iterating the inline comments here too caused two independent
    handlers to triage and post against the same comment in parallel — the
    per-comment lock didn't serialize them because triage takes the lock
    only after a long pre-flight, so both posted clarification replies on
    the same thread.  Closes #518.

    Inline comments are now exclusively handled by the per-comment
    webhook.  This handler is left as a stub so the dispatcher can still
    register it for the event type (and so future top-level review-body
    handling has a place to live).

    Note: the review's *top-level* body text (the box at the bottom of
    \"Submit review\") still arrives only through this event and is not
    yet handled.  Tracked separately — out of scope for the dedup fix.
    """
    _ = (action, config, repo_cfg, gh, already_replied, claude_client, prompts)
    log.debug(
        "reply_to_review: skipping inline comments — handled per-comment (closes #518)"
    )


def needs_more_context(
    comment_body: str, *, claude_client: ProviderAgent | None = None
) -> bool:
    """Ask Haiku whether this comment needs sibling thread context to act on.

    Returns True if Haiku thinks the comment is too vague or cross-referential
    to act on alone (e.g. "same", "ditto", "^"), False otherwise.
    Falls back to False on any error.
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    prompt = (
        f"{NO_TOOLS_CLAUSE}\n\n"
        "A reviewer left this comment on a pull request:\n\n"
        f"{comment_body!r}\n\n"
        "Does this comment need context from sibling review threads to be understood "
        "(e.g. it says 'same', 'ditto', '^', 'here too', or is otherwise too vague "
        "to act on alone)?\n\n"
        "Reply with exactly YES or NO."
    )
    answer = claude_client.print_prompt(prompt, "claude-haiku-4-5").upper()
    return answer.startswith("YES")


_MAX_TITLE_LEN = 80


def _summarize_as_action_item(
    comment_body: str, *, claude_client: ProviderAgent | None = None
) -> str:
    """Ask Opus to convert a comment into a short imperative action-item title.

    If the result is too long, asks Claude to shorten it up to 3 times before
    falling back to hard truncation.
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    prompt = (
        f"{NO_TOOLS_CLAUSE}\n\n"
        "Convert this PR review comment into a short, imperative task title starting with a verb. "
        "Reply with ONLY the title — no category prefix, no punctuation at the end.\n\n"
        f"Comment: {comment_body}"
    )
    result = claude_client.print_prompt(prompt, "claude-opus-4-6").strip()
    for _ in range(3):
        if not result or len(result) <= _MAX_TITLE_LEN:
            break
        result = claude_client.print_prompt(
            f"{NO_TOOLS_CLAUSE}\n\n"
            f"Shorten this task title to under {_MAX_TITLE_LEN} characters while keeping it imperative. "
            f"Reply with ONLY the shortened title.\n\nTitle: {result}",
            "claude-opus-4-6",
        ).strip()
    if not result:
        raise ValueError("_summarize_as_action_item: print_prompt returned empty")
    return result[:_MAX_TITLE_LEN]


def _triage(
    comment_body: str,
    is_bot: bool,
    context: dict[str, Any] | None = None,
    *,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Ask Opus to triage a comment. Returns (category, titles).

    A comment may produce zero or many tasks: titles is a list with one entry
    for ANSWER/ASK/DEFER/DUMP (used as reply context), or one or more entries
    for ACT/DO (each becomes a separate work-queue task).
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    if prompts is None:
        prompts = Prompts("")
    prompt = prompts.triage_prompt(comment_body, is_bot, context)
    log.info("triage classifier: requesting category from opus")
    text = claude_client.print_prompt(prompt, "claude-opus-4-6")
    log.info(
        "triage classifier: returned %d chars (preview=%r)",
        len(text or ""),
        (text or "")[:80],
    )
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
    log.warning(
        "triage classifier: unparseable response, falling back to %s + summarize",
        "DO" if is_bot else "ACT",
    )
    # Fallback: ACT for humans, DO for bots; summarize comment into action item
    category = "DO" if is_bot else "ACT"
    title = _summarize_as_action_item(comment_body, claude_client=claude_client)
    return category, [title]


def reply_to_issue_comment(
    action: Action,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    *,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> tuple[str, list[str]]:
    """Triage and reply to a top-level PR comment (issue_comment event).

    Raises on reply-post failure so callers fail closed.
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    if prompts is None:
        prompts = Prompts(_load_persona(config))
    comment = action.comment_body or ""

    # Extract PR number from prompt
    m = re.search(r"#(\d+)", action.prompt)
    number = m.group(1) if m else ""

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

    category, titles = _triage(
        comment,
        action.is_bot,
        context or None,
        claude_client=claude_client,
        prompts=prompts,
    )
    log.info("issue comment triage: %s — %s", category, titles)

    # For DEFER, open a tracking issue before crafting the reply.
    # Raises on failure so we don't craft a reply referencing a missing issue.
    issue_url: str | None = None
    if category == "DEFER":
        pr_url = f"https://github.com/{repo_full}/pull/{number}" if number else ""
        issue_url = _open_defer_issue(gh, repo_full, pr_url, titles[0], comment)

    instr = prompts.issue_reply_instruction(
        category, comment, ", ".join(titles), action.context, issue_url=issue_url
    )

    log.info("generating %s reply for issue comment on PR #%s", category, number)
    body = claude_client.print_prompt(
        prompts.persona_wrap(instr),
        "claude-opus-4-6",
        system_prompt=prompts.reply_system_prompt(),
    )
    if not body:
        raise ValueError(
            f"issue-comment reply: print_prompt returned empty for PR #{number}"
        )

    log.info("posting issue comment reply on PR #%s: %s", number, body[:80])
    gh.comment_issue(repo_full, number, body)
    log.info("reply posted")

    # Get comment_id from the dispatch payload (stored in context)
    _cid = (action.context or {}).get("comment_id")
    if _cid:
        maybe_react(
            comment,
            _cid,
            "issues",
            repo_full,
            config,
            gh,
            claude_client=claude_client,
            prompts=prompts,
        )

    return (category, titles)


_TYPE_PRIORITY = {TaskType.CI: 0, TaskType.THREAD: 1, TaskType.SPEC: 2}


def _maybe_abort_for_new_task(
    repo_cfg: RepoConfig,
    new_task: dict[str, Any],
    registry: WorkerRegistry,
    *,
    _state: Any = None,
    _tasks: Any = None,
) -> None:
    """Abort the current task if the new task has higher priority.

    Priority is deterministic by type: ci > thread > spec.
    A higher-priority task always preempts — the current task is kept
    pending for later (ABORT_KEEP).  Equal or lower priority does not
    preempt.
    """
    from kennel.state import State
    from kennel.tasks import Tasks

    if _state is None:
        _state = State(repo_cfg.work_dir / ".git" / "fido")
    if _tasks is None:
        _tasks = Tasks(repo_cfg.work_dir)

    state = _state.load()
    current_task_id = state.get("current_task_id")
    if not current_task_id:
        return

    task_list = _tasks.list()
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

    Best-effort enrichment: used to give Opus context about what has already
    been implemented when it reorders the pending task list.  Returns an empty
    string on nonzero exit, subprocess error, or missing git binary — callers
    must not treat the result as authoritative.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except subprocess.SubprocessError, OSError:
        return ""


def _notify_thread_change(
    change: dict[str, Any],
    config: Config,
    gh: GitHub,
    *,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
) -> None:
    """Post a brief comment notifying a commenter that their task was rescoped.

    Called for each thread task that was dropped or modified during dependency
    analysis.  Uses Opus (in Fido's voice) to generate the message; falls back
    to a plain factual note if Opus returns nothing.

    For review comments (comment_type='pulls') replies in-thread via the pull
    review comment API; for issue comments (comment_type='issues') posts via
    the issue comments API.
    """
    if claude_client is None:
        claude_client = ClaudeClient()
    if prompts is None:
        prompts = Prompts(_load_persona(config))

    task = change["task"]
    thread = task.get("thread") or {}
    comment_id = thread.get("comment_id")
    repo = thread.get("repo", "")
    pr = thread.get("pr")
    url = thread.get("url", "")
    author = thread.get("author", "")
    comment_type = thread.get("comment_type", "issues")
    if not (comment_id and repo and pr):
        return

    kind = change["kind"]
    original_title = task.get("title", "")

    if kind == "completed":
        instruction = (
            f"A task originating from a PR comment has been marked done — it was "
            f"covered by work already committed and is no longer in the active queue.\n\n"
            f"Original task: {original_title}\n"
            f"Comment author: {author or '(unknown)'}\n"
            f"Comment: {url}\n\n"
            "Write a very brief reply notifying the commenter that their task has been "
            "marked done because it was covered by recent commits. Reference the comment URL."
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
            "Write a very brief reply notifying the commenter that their original task "
            "has been updated. Reference the comment URL."
        )

    body = claude_client.print_prompt(
        prompts.persona_wrap(instruction),
        "claude-opus-4-6",
        system_prompt=prompts.reply_system_prompt(),
    )
    if not body:
        raise ValueError(
            f"_notify_thread_change: print_prompt returned empty for comment {comment_id}"
        )

    try:
        if comment_type == "pulls":
            gh.reply_to_review_comment(repo, pr, body, comment_id)
        else:
            gh.comment_issue(repo, pr, body)
        log.info("notified thread %s (%s)", comment_id, kind)
    except Exception:
        log.exception("failed to notify thread %s", comment_id)


def _task_snapshot(task_list: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Summarise task_list as an ordered list of (id, status, title) tuples.

    Used by :func:`_rewrite_pr_description` to detect whether the task list
    changed while Opus was generating the PR description.
    """
    return [(t["id"], t.get("status", ""), t.get("title", "")) for t in task_list]


def _rewrite_pr_description(
    work_dir: Path,
    gh: Any,
    *,
    claude_client: ProviderAgent | None = None,
    _state: Any = None,
    _tasks: Any = None,
    _max_retries: int = 3,
) -> None:
    """Rewrite the PR description summary after a successful rescope.

    Delegates to :func:`kennel.worker._write_pr_description` so that initial
    PR creation and post-rescope rewrites share one code path.

    Silently skips when there is no active issue or no open PR for it.
    All other errors (missing ``---`` divider, empty Opus output, GitHub API
    failures) propagate so the caller's thread excepthook can surface them.

    Retries up to *_max_retries* times when the task list changes while Opus
    is generating the description, so the written description always reflects
    the state of the task list at the moment Opus returned.  The PR body is
    re-fetched on each retry so the work-queue section stays current.
    """
    from kennel.state import State
    from kennel.tasks import Tasks
    from kennel.worker import (
        _write_pr_description,  # pyright: ignore[reportPrivateUsage]
    )

    if _state is None:
        _state = State(work_dir / ".git" / "fido")
    if _tasks is None:
        _tasks = Tasks(work_dir)

    state = _state.load()
    issue = state.get("issue")
    if not issue:
        log.info("_rewrite_pr_description: no active issue in state — skipping")
        return

    repo = gh.get_repo_info(cwd=work_dir)
    user = gh.get_user()

    pr_data = gh.find_pr(repo, issue, user)
    if pr_data is None or pr_data.get("state") != "OPEN":
        log.info("_rewrite_pr_description: no open PR for issue #%s — skipping", issue)
        return

    pr_number = pr_data["number"]

    for attempt in range(_max_retries):
        task_list = _tasks.list()
        snapshot_before = _task_snapshot(task_list)

        body = gh.get_pr_body(repo, pr_number)
        _write_pr_description(
            gh,
            repo,
            pr_number,
            issue,
            task_list,
            body,
            claude_client=claude_client,
        )

        snapshot_after = _task_snapshot(_tasks.list())
        if snapshot_after == snapshot_before:
            return

        log.info(
            "_rewrite_pr_description: task list changed during rewrite — retrying"
            " (attempt %d/%d)",
            attempt + 1,
            _max_retries,
        )

    log.warning(
        "_rewrite_pr_description: task list still changing after %d attempts"
        " — description may be slightly stale",
        _max_retries,
    )


def _make_reorder_kwargs(
    work_dir: Path,
    config: Config,
    repo_cfg: RepoConfig | None,
    registry: WorkerRegistry | None,
    gh: Any,
    claude_client: ProviderAgent | None,
    prompts: Prompts | None,
    rewrite_fn: Any,
) -> dict[str, Any]:
    """Build the kwargs dict for a :func:`~kennel.tasks.reorder_tasks` call."""

    def on_changes(changes: list[dict[str, Any]]) -> None:
        for change in changes:
            _notify_thread_change(
                change, config, gh, claude_client=claude_client, prompts=prompts
            )

    def on_done() -> None:
        rewrite_fn(work_dir, gh, claude_client=claude_client)

    kwargs: dict[str, Any] = {
        "_on_changes": on_changes,
        "_on_done": on_done,
        "claude_client": claude_client,
        "prompts": prompts,
    }
    if registry is not None and repo_cfg is not None:

        def on_inprogress_affected() -> None:
            log.info(
                "reorder_tasks_background: in-progress task affected — aborting %s",
                repo_cfg.name,
            )
            registry.abort_task(repo_cfg.name)

        kwargs["_on_inprogress_affected"] = on_inprogress_affected
    return kwargs


def _reorder_tasks_background(
    work_dir: Path,
    commit_summary: str,
    config: Config,
    gh: GitHub,
    repo_cfg: RepoConfig | None = None,
    registry: WorkerRegistry | None = None,
    *,
    _start: Callable[[threading.Thread], None] = threading.Thread.start,
    claude_client: ProviderAgent | None = None,
    prompts: Prompts | None = None,
    _rewrite_fn: Callable[..., None] | None = None,
    _reorder_fn: Callable[..., None] | None = None,
    _coalesce_state: dict[str, Any] | None = None,
) -> None:
    """Run :func:`~kennel.tasks.reorder_tasks` in a daemon background thread.

    Coalesces concurrent calls: if a reorder thread is already running for
    *work_dir*, the new trigger is recorded as pending rather than spawning
    another thread.  When the running thread finishes it checks for a pending
    run and, if one exists, executes it before exiting — so at most one Opus
    call is in-flight plus one queued per repo.

    Passes an ``_on_changes`` callback so that any thread tasks dropped or
    modified during rescoping trigger a notification reply to the original
    comment.

    If *repo_cfg* and *registry* are provided, also passes an
    ``_on_inprogress_affected`` callback that aborts the running worker whenever
    the in-progress task is dropped or modified by the rescope, so the worker
    loop restarts on the new next task.

    Passes an ``_on_done`` callback that rewrites the PR description after a
    successful reorder, so the human-facing summary stays in sync with the
    updated plan.
    """
    from kennel.tasks import reorder_tasks as _reorder_tasks

    reorder = _reorder_fn if _reorder_fn is not None else _reorder_tasks
    rewrite_fn = _rewrite_fn if _rewrite_fn is not None else _rewrite_pr_description
    state = _coalesce_state if _coalesce_state is not None else _reorder_coalesce

    key = str(work_dir)
    kwargs = _make_reorder_kwargs(
        work_dir, config, repo_cfg, registry, gh, claude_client, prompts, rewrite_fn
    )

    with _reorder_coalesce_lock:
        entry = state.setdefault(key, {"running": False, "pending": None})
        if entry["running"]:
            # Coalesce: latest call wins; the running thread will do one more pass.
            entry["pending"] = (commit_summary, kwargs)
            return
        entry["running"] = True
        entry["pending"] = None

    def run_loop() -> None:
        cs = commit_summary
        kw = kwargs
        if repo_cfg is not None:
            set_thread_repo(repo_cfg.name)
        if registry is not None and repo_cfg is not None:
            registry.set_rescoping(repo_cfg.name, True)
        try:
            while True:
                reorder(work_dir, cs, **kw)
                with _reorder_coalesce_lock:
                    pending = state[key].get("pending")
                    if pending is None:
                        state[key]["running"] = False
                        return
                    state[key]["pending"] = None
                    cs, kw = pending
        finally:
            if registry is not None and repo_cfg is not None:
                registry.set_rescoping(repo_cfg.name, False)
            if repo_cfg is not None:
                set_thread_repo(None)

    t = threading.Thread(
        target=run_loop,
        name=f"reorder-{work_dir.name}",
        daemon=True,
    )
    _start(t)


def create_task(
    prompt: str,
    config: Config,
    repo_cfg: RepoConfig,
    gh: GitHub,
    thread: dict[str, Any] | None = None,
    registry: WorkerRegistry | None = None,
    *,
    _get_commit_summary_fn: Callable[[Path], str] = _get_commit_summary,
    _reorder_background_fn: Callable[..., None] = _reorder_tasks_background,
    _tasks: Tasks | None = None,
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
    if _tasks is None:
        _tasks = Tasks(repo_cfg.work_dir)
    # Race guard for thread tasks (#520): if the originating review thread
    # has already been resolved on GitHub (most often because fido completed
    # an earlier task in the same thread and auto-resolved it before this
    # late-arriving triage queued the new task), don't queue.  Without this
    # check the worker re-does work that's already shipped (#521), or
    # rejects the resolved-thread state and reopens it.
    if thread and thread.get("repo") and thread.get("pr") and thread.get("comment_id"):
        try:
            already = gh.is_thread_resolved_for_comment(
                thread["repo"], int(thread["pr"]), int(thread["comment_id"])
            )
        except Exception:
            log.exception(
                "create_task: thread-resolved check failed for comment %s; queuing anyway",
                thread.get("comment_id"),
            )
            already = False
        # Strict ``is True`` so MagicMock test gh stubs (whose method calls
        # return another MagicMock — truthy by default) don't cause this
        # guard to swallow every test-level task creation.  Real GitHub
        # always returns a real bool from ``is_thread_resolved_for_comment``.
        if already is True:
            log.info(
                "create_task: thread for comment %s already resolved on GitHub — "
                "skipping queue (closes #520)",
                thread["comment_id"],
            )
            # Return a synthetic task-shaped dict so callers that don't check
            # status don't choke; callers that DO walk tasks.json won't see it.
            return {
                "title": prompt,
                "type": (TaskType.THREAD if thread else TaskType.SPEC).value,
                "status": "skipped_resolved",
                "thread": thread,
            }
    task_type = TaskType.THREAD if thread else TaskType.SPEC
    log.info("creating task: %s", prompt[:100])
    new_task = _tasks.add(title=prompt, task_type=task_type, thread=thread)
    launch_sync(config, repo_cfg, gh)
    if thread:
        commit_summary = _get_commit_summary_fn(repo_cfg.work_dir)
        _reorder_background_fn(
            repo_cfg.work_dir, commit_summary, config, gh, repo_cfg, registry
        )
    if registry is not None:
        _maybe_abort_for_new_task(repo_cfg, new_task, registry)
    return new_task


def launch_sync(config: Config, repo_cfg: RepoConfig, gh: GitHub) -> None:
    """Sync tasks.json → PR body in a background thread."""
    from kennel.tasks import sync_tasks_background

    sync_tasks_background(repo_cfg.work_dir, gh)
    log.info("sync-tasks launched")


def launch_worker(repo_cfg: RepoConfig, registry: WorkerRegistry) -> None:
    """Wake the per-repo WorkerThread via the registry."""
    log.info("waking worker thread for %s", repo_cfg.name)
    registry.wake(repo_cfg.name)
