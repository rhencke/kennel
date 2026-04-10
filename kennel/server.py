from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from kennel.config import Config, RepoConfig, RepoMembership
from kennel.events import (
    create_task,
    dispatch,
    launch_worker,
    reply_to_comment,
    reply_to_issue_comment,
    reply_to_review,
)
from kennel.github import GitHub
from kennel.registry import WorkerRegistry, make_registry
from kennel.worker import RepoContextFilter, RepoNameFilter

log = logging.getLogger(__name__)


_replied_comments: set[int] = set()

# Exponential backoff for git pull during self-restart: 10s, 30s, 60s
# with a 10-minute total budget. Retries stop once the cumulative delay
# exceeds _PULL_BUDGET_SECONDS, even if a retry window remains.
_PULL_BACKOFF_DELAYS: tuple[int, ...] = (10, 30, 60)
_PULL_BUDGET_SECONDS: float = 600.0


def _runner_dir() -> Path:
    """Return the runner clone directory — where the running kennel code lives."""
    return Path(__file__).resolve().parents[1]


def _get_self_repo(
    runner_dir: Path,
    *,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str | None:
    """Return 'owner/repo' from the runner clone's origin remote, or None on error.

    Handles both SSH (``git@github.com:owner/repo.git``) and HTTPS
    (``https://github.com/owner/repo.git``) remote URLs.
    """
    try:
        result = _run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(runner_dir),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.error("self-restart: failed to read origin remote: %s", e)
        return None
    url = result.stdout.strip()
    m = re.search(r"[:/]([^:/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        log.error("self-restart: could not parse owner/repo from remote url: %r", url)
        return None
    return m.group(1)


def _pull_with_backoff(
    runner_dir: Path,
    *,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    _sleep: Callable[[float], None] = time.sleep,
    _monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Run ``git pull`` in *runner_dir* with exponential backoff on failure.

    Retries with delays from :data:`_PULL_BACKOFF_DELAYS` (10s, 30s, 60s)
    and a total budget of :data:`_PULL_BUDGET_SECONDS` (10 minutes).  Logs
    each attempt and the elapsed time.  Returns ``True`` on success,
    ``False`` if every retry fails or the budget is exhausted.
    """
    start = _monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            _run(
                ["git", "pull"],
                cwd=str(runner_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            log.info(
                "self-restart: git pull succeeded on attempt %d (%.1fs elapsed)",
                attempt,
                _monotonic() - start,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            elapsed = _monotonic() - start
            log.error(
                "self-restart: git pull attempt %d failed after %.1fs: %s",
                attempt,
                elapsed,
                e,
            )
            if attempt > len(_PULL_BACKOFF_DELAYS):
                log.error(
                    "self-restart: git pull exhausted %d retries in %.1fs — giving up",
                    attempt,
                    elapsed,
                )
                return False
            delay = _PULL_BACKOFF_DELAYS[attempt - 1]
            if elapsed + delay > _PULL_BUDGET_SECONDS:
                log.error(
                    "self-restart: git pull would exceed %.0fs budget — giving up",
                    _PULL_BUDGET_SECONDS,
                )
                return False
            log.info("self-restart: sleeping %ds before retry", delay)
            _sleep(delay)


class WebhookHandler(BaseHTTPRequestHandler):
    config: Config
    registry: WorkerRegistry
    # Injectable callables — set as class attributes so HTTP-driven tests can
    # replace them without patching module-level names.
    _fn_reply_to_comment = reply_to_comment
    _fn_reply_to_review = reply_to_review
    _fn_reply_to_issue_comment = reply_to_issue_comment
    _fn_create_task = create_task
    _fn_launch_worker = launch_worker
    _fn_runner_dir = _runner_dir
    _fn_get_self_repo = _get_self_repo
    _fn_pull_with_backoff = _pull_with_backoff
    _fn_os_chdir = os.chdir
    _fn_os_execvp = os.execvp

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._respond(400, "empty body")
            return

        body = self.rfile.read(content_length)

        if not self._verify_signature(body):
            log.warning(
                "signature verification failed — %s %s",
                self.headers.get("X-GitHub-Event", "?"),
                self.client_address[0],
            )
            self._respond(401, "bad signature")
            return

        event = self.headers.get("X-GitHub-Event", "")
        delivery = self.headers.get("X-GitHub-Delivery", "?")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, "invalid json")
            return

        # Route by repo
        repo_name = payload.get("repository", {}).get("full_name", "")
        repo_cfg = self.config.repos.get(repo_name)

        log.info(
            "webhook: event=%s action=%s repo=%s delivery=%s",
            event,
            payload.get("action", "-"),
            repo_name,
            delivery,
        )

        # Respond immediately — don't block on dispatch
        self._respond(200, "ok")

        # Check for self-restart (kennel repo merged).  _self_restart itself
        # verifies via the runner clone's git remote that the webhook's repo
        # actually matches kennel — if so it execs a new process and never
        # returns.  For other merged PRs (e.g. fido's own work) it's a no-op.
        if (
            event == "pull_request"
            and payload.get("action") == "closed"
            and payload.get("pull_request", {}).get("merged")
        ):
            self._self_restart(repo_name)

        if not repo_cfg:
            log.debug("ignoring webhook for unregistered repo: %s", repo_name)
            return

        # Process in background thread so we don't block the server
        action = dispatch(event, payload, self.config, repo_cfg)
        if action:
            threading.Thread(
                target=self._process_action,
                args=(action, repo_cfg),
                daemon=True,
            ).start()

    def _process_action(self, action, repo_cfg: RepoConfig) -> None:
        try:
            handled = False

            if action.reply_to:
                cid = action.reply_to.get("comment_id")
                if cid and cid in _replied_comments:
                    log.info("already replied to comment %s — skipping", cid)
                    handled = True
                    category, titles = None, []
                else:
                    posted, category, titles = type(self)._fn_reply_to_comment(
                        action, self.config, repo_cfg
                    )
                    if cid and posted:
                        _replied_comments.add(cid)
                    handled = True
                # Create task based on triage result.
                # DEFER files a GitHub issue (handled in reply_to_comment) — no tasks.json entry.
                # ACT, DO → add each task title to work queue.
                if category not in ("DUMP", "ANSWER", "ASK", "DEFER"):
                    for title in titles or []:
                        type(self)._fn_create_task(
                            title,
                            self.config,
                            repo_cfg,
                            thread=action.reply_to,
                            registry=self.registry,
                        )

            if action.review_comments:
                type(self)._fn_reply_to_review(
                    action, self.config, repo_cfg, already_replied=_replied_comments
                )
                handled = True  # inline comments handled individually

            # Top-level PR comments (issue_comment) — no reply_to, but has comment_body
            if not handled and action.comment_body:
                category, titles = type(self)._fn_reply_to_issue_comment(
                    action, self.config, repo_cfg
                )
                handled = True
                # DEFER files a GitHub issue — no tasks.json entry.
                if category not in ("DUMP", "ANSWER", "ASK", "DEFER"):
                    for title in titles:
                        type(self)._fn_create_task(
                            title,
                            self.config,
                            repo_cfg,
                            thread=action.thread,
                            registry=self.registry,
                        )

            # Non-comment events just trigger kennel worker — no task needed
            type(self)._fn_launch_worker(repo_cfg, self.registry)
        except Exception:
            log.exception("error processing action")

    def _self_restart(self, repo_name: str) -> None:
        runner_dir = type(self)._fn_runner_dir()
        self_repo = type(self)._fn_get_self_repo(runner_dir)
        if self_repo != repo_name:
            return  # Not our repo — nothing to do.
        log.info(
            "kennel repo %s merged — pulling runner clone at %s",
            repo_name,
            runner_dir,
        )
        self.registry.stop_and_join(repo_name)
        if not type(self)._fn_pull_with_backoff(runner_dir):
            log.error("self-restart: git pull gave up — running old version")
            return
        type(self)._fn_os_chdir(runner_dir)
        type(self)._fn_os_execvp("uv", ["uv", "run", "kennel", *sys.argv[1:]])

    def do_GET(self) -> None:
        if self.path == "/status":
            activities = [
                {"repo_name": a.repo_name, "what": a.what, "busy": a.busy}
                for a in self.registry.get_all_activities()
            ]
            body = json.dumps(activities).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._respond(200, "kennel is running")

    def _verify_signature(self, body: bytes) -> bool:
        header = self.headers.get("X-Hub-Signature-256", "")
        if not header:
            return False
        expected = (
            "sha256=" + hmac.new(self.config.secret, body, hashlib.sha256).hexdigest()
        )
        return hmac.compare_digest(expected, header)

    def _respond(self, code: int, message: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())

    def log_message(self, format: str, *args: object) -> None:
        pass


def populate_memberships(
    config: Config, *, _gh_factory: Callable[[], GitHub] = GitHub
) -> None:
    """Fetch collaborators for each repo once at startup and store on RepoConfig.

    Mutates ``config.repos`` in place — each :class:`RepoConfig` is replaced
    with a new instance carrying a populated :class:`RepoMembership`.  Uses
    one GitHub client instance for all repos.  Bot account (gh_user) is
    excluded from every collaborator set.
    """
    gh = _gh_factory()
    bot_user = gh.get_user()
    for name, repo_cfg in list(config.repos.items()):
        collabs = frozenset(c for c in gh.get_collaborators(name) if c != bot_user)
        log.info("%s: collaborators = %s", name, sorted(collabs) or "(none)")
        config.repos[name] = dataclasses.replace(
            repo_cfg, membership=RepoMembership(collaborators=collabs)
        )


def run(
    *,
    _from_args=Config.from_args,
    _HTTPServer=HTTPServer,
    _make_registry=make_registry,
    _path_home=Path.home,
    _basic_config=logging.basicConfig,
    _stderr=sys.stderr,
    _populate_memberships=populate_memberships,
) -> None:
    config = _from_args()

    log_file = _path_home() / "log" / "kennel.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    repo_filter = RepoContextFilter()
    handlers: list[logging.Handler] = [logging.FileHandler(log_file)]
    if _stderr.isatty():
        handlers.append(logging.StreamHandler(_stderr))
    for handler in handlers:
        handler.addFilter(repo_filter)

    for repo_full_name in config.repos:
        short_name = repo_full_name.split("/")[-1]
        repo_log = log_file.parent / f"kennel-{short_name}.log"
        repo_handler = logging.FileHandler(repo_log)
        repo_handler.addFilter(repo_filter)
        repo_handler.addFilter(RepoNameFilter(short_name))
        handlers.append(repo_handler)

    _basic_config(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(repo_name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    _populate_memberships(config)

    WebhookHandler.config = config
    WebhookHandler.registry = _make_registry(config.repos)

    server = _HTTPServer(("", config.port), WebhookHandler)
    repos_str = ", ".join(f"{name}={rc.work_dir}" for name, rc in config.repos.items())
    log.info("kennel listening on :%d — repos: %s", config.port, repos_str)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()
