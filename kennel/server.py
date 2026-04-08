from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from kennel.config import Config, RepoConfig
from kennel.events import (
    create_task,
    dispatch,
    launch_worker,
    reply_to_comment,
    reply_to_issue_comment,
    reply_to_review,
)
from kennel.registry import WorkerRegistry, make_registry

log = logging.getLogger(__name__)


_replied_comments: set[int] = set()


class WebhookHandler(BaseHTTPRequestHandler):
    config: Config
    registry: WorkerRegistry

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

        # Check for self-restart (kennel repo merged)
        if (
            self.config.self_repo
            and repo_name == self.config.self_repo
            and event == "pull_request"
            and payload.get("action") == "closed"
            and payload.get("pull_request", {}).get("merged")
        ):
            self._self_restart(repo_name)
            return

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
                    category, title = None, None
                else:
                    posted, category, title = reply_to_comment(
                        action, self.config, repo_cfg
                    )
                    if cid and posted:
                        _replied_comments.add(cid)
                    handled = True
                # Create task based on triage result.
                # DEFER files a GitHub issue (handled in reply_to_comment) — no tasks.json entry.
                # ACT, DO → add to work queue.
                if category in ("DUMP", "ANSWER", "ASK", "DEFER"):
                    pass  # No task needed
                elif title:
                    create_task(
                        title,
                        self.config,
                        repo_cfg,
                        thread=action.reply_to,
                    )

            if action.review_comments:
                reply_to_review(
                    action, self.config, repo_cfg, already_replied=_replied_comments
                )
                handled = True  # inline comments handled individually

            # Top-level PR comments (issue_comment) — no reply_to, but has comment_body
            if not handled and action.comment_body:
                category, title = reply_to_issue_comment(action, self.config, repo_cfg)
                handled = True
                # DEFER files a GitHub issue — no tasks.json entry.
                if category not in ("DUMP", "ANSWER", "ASK", "DEFER") and title:
                    create_task(title, self.config, repo_cfg)

            # Non-comment events just trigger kennel worker — no task needed
            launch_worker(repo_cfg, self.registry)
        except Exception:
            log.exception("error processing action")

    def _self_restart(self, repo_name: str) -> None:
        repo_cfg = self.config.repos.get(repo_name)
        if repo_cfg:
            log.info("kennel repo %s merged — pulling and restarting", repo_name)
            subprocess.run(
                ["git", "reset", "--hard"],
                cwd=str(repo_cfg.work_dir),
                capture_output=True,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=str(repo_cfg.work_dir),
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", "main"],
                cwd=str(repo_cfg.work_dir),
                capture_output=True,
            )
            subprocess.run(
                ["git", "pull"], cwd=str(repo_cfg.work_dir), capture_output=True
            )
            os.execv(sys.argv[0], sys.argv)

    def do_GET(self) -> None:
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


def run() -> None:
    config = Config.from_args()

    log_file = Path.home() / "log" / "kennel.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.FileHandler(log_file)]
    if sys.stderr.isatty():
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    WebhookHandler.config = config
    WebhookHandler.registry = make_registry(config.repos)

    server = HTTPServer(("", config.port), WebhookHandler)
    repos_str = ", ".join(f"{name}={rc.work_dir}" for name, rc in config.repos.items())
    log.info("kennel listening on :%d — repos: %s", config.port, repos_str)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()
