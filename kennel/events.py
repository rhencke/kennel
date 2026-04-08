from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kennel.config import Config, RepoConfig
from kennel.github import get_github
from kennel.prompts import (
    Prompts,
    issue_reply_instruction,
    reply_instruction,
    triage_prompt,
)
from kennel.tasks import add_task

log = logging.getLogger(__name__)


@dataclass
class Action:
    prompt: str
    reply_to: dict[str, Any] | None = None  # {repo, pr, comment_id}
    review_comments: dict[str, Any] | None = None  # {repo, pr, review_id}
    comment_body: str | None = None
    is_bot: bool = False
    context: dict[str, Any] | None = None  # {pr_title, file, diff_hunk, line, pr_body}


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
                "pr_body": (pr.get("body", "") or "")[:500],
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
                "pr_body": (issue.get("body", "") or "")[:500],
                "comment_id": comment_id,
            },
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
) -> None:
    """Let Fido decide whether to react to a comment with an emoji.

    comment_type: 'pulls' for review comments, 'issues' for issue comments.
    """
    persona_path = config.sub_dir / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    prompts = Prompts(persona)
    try:
        result = subprocess.run(
            [
                "claude",
                "--model",
                "claude-opus-4-6",
                "--print",
                "-p",
                prompts.react_prompt(comment_body),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        reaction = (
            result.stdout.strip().lower().split("\n")[0].strip()
            if result.returncode == 0
            else ""
        )
    except subprocess.TimeoutExpired, FileNotFoundError:
        return

    valid = {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
    if reaction not in valid:
        log.debug("fido chose not to react (got: %s)", reaction)
        return

    log.info("fido reacts with %s to comment %s", reaction, comment_id)
    try:
        get_github().add_reaction(repo, comment_type, comment_id, reaction)
    except Exception:
        log.exception("failed to post reaction")


def reply_to_comment(
    action: Action, config: Config, repo_cfg: RepoConfig
) -> tuple[bool, str, str]:
    """Triage a comment via Opus, generate a reply via Opus, post it.

    Returns (posted, triage_category, task_title).
    posted is True only when the reply was successfully sent to GitHub.
    Uses a per-comment lockfile to prevent races with work.sh.
    """
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

    persona_path = config.sub_dir / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    prompts = Prompts(persona)
    comment = action.comment_body

    # Enrich context with sibling threads when the comment needs more context
    context: dict[str, Any] = dict(action.context) if action.context else {}
    if needs_more_context(comment) and info.get("repo") and info.get("pr"):
        siblings = get_github().fetch_sibling_threads(info["repo"], info["pr"])
        if siblings:
            context["sibling_threads"] = siblings
            log.info(
                "needs-more-context comment — fetched %d sibling thread(s) for context",
                len(siblings),
            )

    # Step 1: Haiku triage
    category, title = _triage(comment, action.is_bot, context)
    log.info("triage: %s — %s", category, title)

    # Step 2: For DEFER, open a tracking issue before crafting the reply
    issue_url: str | None = None
    if category == "DEFER" and info.get("repo"):
        pr_url = f"https://github.com/{info['repo']}/pull/{info['pr']}"
        issue_body = f"Deferred from {pr_url}\n\n> {comment}"
        try:
            issue_url = get_github().create_issue(info["repo"], title, issue_body)
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
    try:
        result = subprocess.run(
            [
                "claude",
                "--model",
                "claude-opus-4-6",
                "--print",
                "-p",
                prompts.persona_wrap(instr),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        body = result.stdout.strip() if result.returncode == 0 else ""
    except subprocess.TimeoutExpired, FileNotFoundError:
        body = ""

    if not body:
        body = (
            "Looking into this now."
            if category in ("ACT", "DO")
            else "Noted — checking on this."
        )

    log.info("posting reply to PR #%s: %s", info["pr"], body[:80])
    posted = False
    try:
        get_github().reply_to_review_comment(
            info["repo"], info["pr"], body, info["comment_id"]
        )
        posted = True
        log.info("reply posted")
    except Exception:
        log.exception("failed to post reply")

    # Maybe react
    maybe_react(comment, info["comment_id"], "pulls", info.get("repo", ""), config)

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
) -> None:
    """Fetch inline comments from a review and reply to each."""
    info = action.review_comments
    if not info:
        return

    log.info(
        "fetching review comments for PR #%s review %s", info["pr"], info["review_id"]
    )
    try:
        comments = get_github().get_review_comments(
            info["repo"], info["pr"], info["review_id"]
        )
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
        )
        if posted and already_replied is not None:
            already_replied.add(cid)


def needs_more_context(comment_body: str) -> bool:
    """Ask Haiku whether this comment needs sibling thread context to act on.

    Returns True if Haiku thinks the comment is too vague or cross-referential
    to act on alone (e.g. "same", "ditto", "^"), False otherwise.
    Falls back to False on any error.
    """
    prompt = (
        "A reviewer left this comment on a pull request:\n\n"
        f"{comment_body!r}\n\n"
        "Does this comment need context from sibling review threads to be understood "
        "(e.g. it says 'same', 'ditto', '^', 'here too', or is otherwise too vague "
        "to act on alone)?\n\n"
        "Reply with exactly YES or NO."
    )
    try:
        result = subprocess.run(
            ["claude", "--model", "claude-haiku-4-5", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=10,
        )
        answer = result.stdout.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False


def _triage(
    comment_body: str, is_bot: bool, context: dict[str, Any] | None = None
) -> tuple[str, str]:
    """Ask Haiku to triage a comment. Returns (prefix, title)."""
    prompt = triage_prompt(comment_body, is_bot, context)
    try:
        result = subprocess.run(
            ["claude", "--model", "claude-opus-4-6", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=15,
        )
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if ":" in line:
            prefix, title = line.split(":", 1)
            prefix = prefix.strip().upper()
            title = title.strip()
            if prefix in ("ACT", "ASK", "ANSWER", "DO", "DEFER", "DUMP"):
                return prefix, title
    except Exception:
        pass
    # Fallback: ACT for humans, DO for bots
    return ("DO" if is_bot else "ACT"), comment_body[:80]


def reply_to_issue_comment(
    action: Action, config: Config, repo_cfg: RepoConfig
) -> tuple[str, str]:
    """Triage and reply to a top-level PR comment (issue_comment event)."""
    comment = action.comment_body or ""

    # Extract PR number from prompt
    import re

    m = re.search(r"#(\d+)", action.prompt)
    number = m.group(1) if m else ""

    persona_path = config.sub_dir / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    prompts = Prompts(persona)
    category, title = _triage(comment, action.is_bot, action.context)
    log.info("issue comment triage: %s — %s", category, title)

    # For DEFER, open a tracking issue before crafting the reply
    issue_url: str | None = None
    if category == "DEFER":
        repo_full = get_github().get_repo_info(cwd=repo_cfg.work_dir)
        pr_url = f"https://github.com/{repo_full}/pull/{number}" if number else ""
        issue_body = f"Deferred from {pr_url}\n\n> {comment}" if pr_url else comment
        try:
            issue_url = get_github().create_issue(repo_full, title, issue_body)
            log.info("opened tracking issue for DEFER: %s", issue_url)
        except Exception:
            log.exception("failed to open tracking issue for DEFER")

    instr = issue_reply_instruction(
        category, comment, title, action.context, issue_url=issue_url
    )

    log.info("generating %s reply for issue comment on PR #%s", category, number)
    try:
        result = subprocess.run(
            [
                "claude",
                "--model",
                "claude-opus-4-6",
                "--print",
                "-p",
                prompts.persona_wrap(instr),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        body = result.stdout.strip() if result.returncode == 0 else ""
    except subprocess.TimeoutExpired, FileNotFoundError:
        body = ""
    if not body:
        body = "On it!" if category in ("ACT", "DO") else "Noted."

    log.info("posting issue comment reply on PR #%s: %s", number, body[:80])
    try:
        repo_full = get_github().get_repo_info(cwd=repo_cfg.work_dir)
        get_github().comment_issue(repo_full, number, body)
        log.info("reply posted")
    except Exception:
        log.exception("failed to post issue comment reply")

    # Maybe react (extract comment_id from context)
    if "#" in action.prompt:
        import re

        m = re.search(r"#(\d+)", action.prompt)
        number = m.group(1) if m else ""
    # Get comment_id from the dispatch payload (stored in context)
    _cid = (action.context or {}).get("comment_id")
    if _cid:
        repo_full = get_github().get_repo_info(cwd=repo_cfg.work_dir)
        maybe_react(comment, _cid, "issues", repo_full, config)

    return (category, title)


def create_task(
    prompt: str,
    config: Config,
    repo_cfg: RepoConfig,
    thread: dict[str, Any] | None = None,
) -> None:
    """Write a task to the shared task file, then trigger sync.

    PR comment tasks (those with a thread) carry a thread payload that causes
    ``_pick_next_task`` to prioritise them as NEXT (second only to CI failures),
    without inserting them out-of-order in the list.
    """
    log.info("creating task: %s", prompt[:100])
    add_task(repo_cfg.work_dir, title=prompt, thread=thread)
    launch_sync(config, repo_cfg)


def launch_sync(config: Config, repo_cfg: RepoConfig) -> None:
    """Launch sync-tasks.sh in background.

    TODO: remove once sync-tasks.sh is rewritten to Python.
    """
    sync_script = config.sub_dir.parent / "sync-tasks.sh"
    try:
        subprocess.Popen(
            ["bash", str(sync_script), str(repo_cfg.work_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        log.info("sync-tasks launched")
    except Exception:
        log.exception("failed to launch sync-tasks")


def launch_worker(config: Config, repo_cfg: RepoConfig) -> int | None:
    """Launch work.sh in background (disowned). Returns PID.

    TODO: replace with a call to worker.run() once work.sh is fully rewritten to Python.
    """
    work_script = config.sub_dir.parent / "work.sh"
    log_path = repo_cfg.work_dir / ".git" / "fido" / "fido.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("launching work.sh → %s", repo_cfg.work_dir)
    try:
        with open(log_path, "a") as log_file:
            proc = subprocess.Popen(
                ["bash", str(work_script), str(repo_cfg.work_dir)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        log.info("work.sh launched — pid=%d", proc.pid)
        return proc.pid
    except Exception:
        log.exception("failed to launch work.sh")
        return None
