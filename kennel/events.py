from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from kennel.config import Config

log = logging.getLogger("kennel")


def dispatch(event: str, payload: dict[str, Any], config: Config) -> str | None:
    """Map a GitHub webhook event to a task prompt. Returns None if ignored."""
    action = payload.get("action", "")

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
        return f"New issue #{number} assigned to {assignee}: {title}"

    if event == "pull_request_review" and action == "submitted":
        review = payload.get("review", {})
        pr = payload.get("pull_request", {})
        number = pr.get("number")
        state = review.get("state", "")
        user = review.get("user", {}).get("login", "")
        if not number:
            return None
        log.info("review on PR #%s: %s by %s", number, state, user)
        return f"Review on PR #{number}: {state} by {user}"

    if event == "pull_request_review_comment" and action == "created":
        comment = payload.get("comment", {})
        pr = payload.get("pull_request", {})
        number = pr.get("number")
        user = comment.get("user", {}).get("login", "")
        # ignore our own comments
        if user.lower() in ("fidocancode", "fido-can-code"):
            log.debug("ignoring own comment on PR #%s", number)
            return None
        if not number:
            return None
        body_preview = (comment.get("body", "") or "")[:80]
        log.info("comment on PR #%s by %s: %s", number, user, body_preview)
        return f"Review comment on PR #{number} by {user}: {body_preview}"

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
        return f"CI failure on {pr_str}: {name} ({conclusion})"

    if event == "pull_request" and action == "closed":
        pr = payload.get("pull_request", {})
        if not pr.get("merged"):
            log.debug("PR #%s closed without merge — ignoring", pr.get("number"))
            return None
        number = pr.get("number")
        log.info("PR #%s merged", number)
        return f"PR #{number} merged — cleanup"

    log.debug("ignored event: %s (action=%s)", event, action)
    return None


def update_task_list(prompt: str, config: Config) -> None:
    """Run a fresh claude CLI to update the shared task list."""
    env = {**os.environ, "CLAUDE_CODE_TASK_LIST_ID": config.project}
    log.info("updating task list: %s", prompt[:100])
    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
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
