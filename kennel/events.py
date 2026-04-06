from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kennel.config import Config

log = logging.getLogger("kennel")


@dataclass
class Action:
    prompt: str
    reply_to: dict[str, Any] | None = None  # {repo, pr, comment_id}


def _is_allowed(user: str, payload: dict[str, Any], config: Config) -> bool:
    """Check if user is the repo owner or an allowed bot."""
    owner = payload.get("repository", {}).get("owner", {}).get("login", "")
    return user == owner or user in config.allowed_bots


def dispatch(event: str, payload: dict[str, Any], config: Config) -> Action | None:
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
        if not number:
            return None
        if not _is_allowed(user, payload, config):
            log.debug("ignoring review on PR #%s by %s (not allowed)", number, user)
            return None
        log.info("review on PR #%s: %s by %s", number, state, user)
        return Action(prompt=f"Review on PR #{number}: {state} by {user}")

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
        body_preview = (comment.get("body", "") or "")[:80]
        log.info("comment on PR #%s by %s: %s", number, user, body_preview)
        return Action(
            prompt=f"Review comment on PR #{number} by {user}: {body_preview}",
            reply_to={"repo": repo, "pr": number, "comment_id": comment_id},
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


def reply_to_comment(action: Action, config: Config) -> None:
    """Post an immediate acknowledgment reply to a PR comment via Opus."""
    info = action.reply_to
    if not info:
        return

    persona_path = Path(config.work_script).parent / "sub" / "persona.md"
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""

    plain = f"Acknowledged — working on this now."
    log.info("generating reply for PR #%s comment %s", info["pr"], info["comment_id"])
    try:
        result = subprocess.run(
            [
                "claude", "--model", "claude-opus-4-6", "--print", "-p",
                f"{persona}\n\nRewrite the following GitHub PR comment reply in character as Fido. "
                f"Keep it to 1 sentence. Output only the comment text, no quotes, no explanation.\n\n{plain}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        body = result.stdout.strip() if result.returncode == 0 else plain
    except (subprocess.TimeoutExpired, FileNotFoundError):
        body = plain

    if not body:
        body = plain

    log.info("posting reply to PR #%s: %s", info["pr"], body[:80])
    try:
        subprocess.run(
            [
                "gh", "api",
                f"repos/{info['repo']}/pulls/{info['pr']}/comments",
                "-X", "POST",
                "-f", f"body={body}",
                "-F", f"in_reply_to={info['comment_id']}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        log.info("reply posted")
    except Exception:
        log.exception("failed to post reply")


def update_task_list(prompt: str, config: Config) -> None:
    """Run a fresh claude CLI to update the shared task list."""
    env = {**os.environ, "CLAUDE_CODE_TASK_LIST_ID": config.project}
    full_prompt = (
        f"{prompt}\n\n"
        "Add a task for this event. Do NOT edit the PR body, "
        "do NOT post comments, do NOT run any git commands. "
        "Only update the task list."
    )
    log.info("updating task list: %s", prompt[:100])
    try:
        result = subprocess.run(
            ["claude", "--print", "-p", full_prompt],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            log.warning(
                "claude task update failed (exit %d): %s",
                result.returncode,
                result.stderr[:200],
            )
        else:
            log.info("task list updated")
    except subprocess.TimeoutExpired:
        log.warning("claude task update timed out")
    except FileNotFoundError:
        log.error("claude CLI not found — is it on PATH?")


def launch_worker(config: Config) -> int | None:
    """Launch work.sh in background (disowned). Returns PID."""
    env = {**os.environ, "CLAUDE_CODE_TASK_LIST_ID": config.project}
    log_path = config.work_dir / ".git" / "fido" / "fido.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("launching work.sh → %s", config.work_dir)
    try:
        with open(log_path, "a") as log_file:
            proc = subprocess.Popen(
                ["bash", str(config.work_script), str(config.work_dir)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        log.info("work.sh launched — pid=%d", proc.pid)
        return proc.pid
    except Exception:
        log.exception("failed to launch work.sh")
        return None
