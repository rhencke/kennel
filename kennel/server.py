from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from kennel.config import Config
from kennel.events import dispatch, launch_worker, update_task_list

log = logging.getLogger("kennel")


class WebhookHandler(BaseHTTPRequestHandler):
    config: Config

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

        log.info(
            "webhook: event=%s action=%s delivery=%s",
            event,
            payload.get("action", "-"),
            delivery,
        )

        # Respond immediately — don't block on dispatch
        self._respond(200, "ok")

        # Dispatch: update task list + launch worker
        prompt = dispatch(event, payload, self.config)
        if prompt:
            update_task_list(prompt, self.config)
            launch_worker(self.config)

    def do_GET(self) -> None:
        self._respond(200, "kennel is running")

    def _verify_signature(self, body: bytes) -> bool:
        header = self.headers.get("X-Hub-Signature-256", "")
        if not header:
            return False
        expected = "sha256=" + hmac.new(
            self.config.secret, body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, header)

    def _respond(self, code: int, message: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default access log — we log ourselves
        pass


def run() -> None:
    config = Config.from_env()

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    WebhookHandler.config = config

    server = HTTPServer(("", config.port), WebhookHandler)
    log.info(
        "kennel listening on :%d — project=%s work_dir=%s",
        config.port,
        config.project,
        config.work_dir,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()
