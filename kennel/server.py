from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from kennel.claude import kill_active_children
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
from kennel.watchdog import _STALE_THRESHOLD, Watchdog  # noqa: PLC2701
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


def _parse_repo_from_url(url: str) -> str | None:
    """Extract 'owner/repo' from an SSH or HTTPS git remote URL, or return None."""
    m = re.search(r"[:/]([^:/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


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
    parsed = _parse_repo_from_url(url)
    if not parsed:
        log.error("self-restart: could not parse owner/repo from remote url: %r", url)
        return None
    return parsed


def preflight_repo_identity(
    repos: dict[str, RepoConfig],
    *,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Verify each configured work_dir is a git repo whose origin matches its name.

    Raises :exc:`SystemExit` if any repo's origin remote can't be read, can't
    be parsed, or doesn't match the configured ``owner/repo`` name.  This runs
    once at startup so misconfigured repo mappings fail immediately rather than
    surfacing as silent divergence deep inside webhook or worker paths.
    """
    for name, repo_cfg in repos.items():
        try:
            result = _run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repo_cfg.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"preflight: {name}: git remote get-url failed: {e}"
            ) from e
        except FileNotFoundError as e:
            raise SystemExit(f"preflight: {name}: git not found: {e}") from e
        url = result.stdout.strip()
        actual = _parse_repo_from_url(url)
        if actual is None:
            raise SystemExit(
                f"preflight: {name}: could not parse owner/repo from origin remote: {url!r}"
            )
        if actual != name:
            raise SystemExit(
                f"preflight: {name}: origin remote is {actual!r} — expected {name!r}"
            )
        log.info("preflight: %s: work_dir identity confirmed", name)


_REQUIRED_TOOLS = ("git", "gh", "claude")


def preflight_tools(
    *,
    _which: Callable[[str], str | None] = shutil.which,
) -> None:
    """Verify that all required CLI tools are on PATH.

    Raises :exc:`SystemExit` naming the first missing tool.  Runs once at
    startup so a missing binary is caught immediately rather than discovered
    inside a worker or webhook handler hours later.
    """
    for tool in _REQUIRED_TOOLS:
        if _which(tool) is None:
            raise SystemExit(f"preflight: required tool not found on PATH: {tool!r}")
    log.info("preflight: all required tools found: %s", ", ".join(_REQUIRED_TOOLS))


def preflight_sub_dir(
    config: Config,
    *,
    _is_dir: Callable[[Path], bool] = Path.is_dir,
) -> None:
    """Verify that the skill-files directory exists.

    Raises :exc:`SystemExit` if ``config.sub_dir`` is not an existing directory.
    Workers read ``persona.md`` and sub-skill files from here on every task
    run — a missing directory causes every worker invocation to fail with an
    obscure I/O error rather than a clear startup message.
    """
    if not _is_dir(config.sub_dir):
        raise SystemExit(
            f"preflight: skill-files directory not found: {config.sub_dir}"
        )
    log.info("preflight: skill-files directory confirmed: %s", config.sub_dir)


def preflight_gh_auth(
    *,
    _gh_factory: Callable[[], GitHub] = GitHub,
) -> None:
    """Verify gh auth works by fetching the authenticated bot user.

    Raises :exc:`SystemExit` if the GitHub client cannot be constructed or
    ``get_user()`` fails for any reason (bad token, network error, etc.).
    Runs once at startup so auth failures surface immediately rather than
    deep inside a webhook or worker path.
    """
    try:
        bot_user = _gh_factory().get_user()
    except Exception as e:
        raise SystemExit(f"preflight: gh auth check failed: {e}") from e
    log.info("preflight: gh auth confirmed — bot user is %r", bot_user)


def _get_head(
    runner_dir: Path,
    *,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str | None:
    """Return the current HEAD commit hash of the runner clone, or None on error."""
    try:
        result = _run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(runner_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return None


def _pull_with_backoff(
    runner_dir: Path,
    *,
    _run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    _sleep: Callable[[float], None] = time.sleep,
    _monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Sync the runner clone to ``origin/main`` with exponential backoff.

    Uses ``git fetch origin main`` followed by ``git reset --hard origin/main``
    so the runner clone is forcibly synced to the remote — no merge strategy
    needed, no failure on local divergence.  The runner clone is supposed to
    be read-only (fido never commits there), so a destructive reset is safe
    and the only way to recover from accidental local commits or detached
    state.

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
                ["git", "fetch", "origin", "main"],
                cwd=str(runner_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            _run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=str(runner_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            log.info(
                "self-restart: runner synced on attempt %d (%.1fs elapsed)",
                attempt,
                _monotonic() - start,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            elapsed = _monotonic() - start
            log.error(
                "self-restart: runner sync attempt %d failed after %.1fs: %s",
                attempt,
                elapsed,
                e,
            )
            if attempt > len(_PULL_BACKOFF_DELAYS):
                log.error(
                    "self-restart: runner sync exhausted %d retries in %.1fs — giving up",
                    attempt,
                    elapsed,
                )
                return False
            delay = _PULL_BACKOFF_DELAYS[attempt - 1]
            if elapsed + delay > _PULL_BUDGET_SECONDS:
                log.error(
                    "self-restart: runner sync would exceed %.0fs budget — giving up",
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
    _fn_get_github = GitHub
    _fn_dispatch = dispatch
    _fn_reply_to_comment = reply_to_comment
    _fn_reply_to_review = reply_to_review
    _fn_reply_to_issue_comment = reply_to_issue_comment
    _fn_create_task = create_task
    _fn_launch_worker = launch_worker
    _fn_runner_dir = _runner_dir
    _fn_get_self_repo = _get_self_repo
    _fn_get_head = _get_head
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

        # Pre-compute self-restart triggers — needed for both registered and
        # unregistered repos (_self_restart verifies the runner's git remote).
        default_branch = payload.get("repository", {}).get("default_branch", "")
        is_pr_merged = (
            event == "pull_request"
            and payload.get("action") == "closed"
            and bool(payload.get("pull_request", {}).get("merged"))
        )
        is_default_push = (
            event == "push"
            and bool(default_branch)
            and payload.get("ref") == f"refs/heads/{default_branch}"
        )

        if not repo_cfg:
            # Nothing to dispatch — ack immediately, then maybe self-restart.
            self._respond(200, "ok")
            if is_pr_merged:
                self._self_restart(repo_name, reason="PR merged")
            elif is_default_push:
                self._self_restart(repo_name, reason=f"push to {default_branch}")
            else:
                log.debug("ignoring webhook for unregistered repo: %s", repo_name)
            return

        # Dispatch BEFORE acknowledging — if dispatch raises, return 500 so
        # GitHub retries instead of treating the event as successfully handled.
        try:
            action = type(self)._fn_dispatch(event, payload, self.config, repo_cfg)
        except Exception:
            log.exception("dispatch failed for %s", repo_name)
            self._respond(500, "dispatch error")
            return

        # Acknowledge only after dispatch succeeds.
        self._respond(200, "ok")

        # Self-restart after ack so the response reaches GitHub before exec.
        if is_pr_merged:
            self._self_restart(repo_name, reason="PR merged")
        elif is_default_push:
            self._self_restart(repo_name, reason=f"push to {default_branch}")

        # Process in background thread so we don't block the server.
        if action:
            threading.Thread(
                target=self._process_action,
                args=(action, repo_cfg),
                daemon=True,
            ).start()

    def _process_action(self, action, repo_cfg: RepoConfig) -> None:
        try:
            self.registry.report_activity(
                repo_cfg.name, "handling webhook action", busy=True
            )
            handled = False

            if action.reply_to:
                cid = action.reply_to.get("comment_id")
                if cid and cid in _replied_comments:
                    log.info("already replied to comment %s — skipping", cid)
                    handled = True
                    category, titles = None, []
                else:
                    category, titles = type(self)._fn_reply_to_comment(
                        action, self.config, repo_cfg
                    )
                    if cid:
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
            self._signal_action_error(action)

    def _signal_action_error(self, action) -> None:
        """Post a 'confused' reaction on the triggering comment, if any.

        Called when _process_action raises so the comment author sees something
        went wrong rather than silence.  Reaction failures are caught — if
        signalling itself fails we log and move on rather than masking the
        original error.
        """
        thread = action.reply_to or action.thread
        if not thread:
            return
        repo = thread.get("repo")
        comment_id = thread.get("comment_id")
        comment_type = thread.get("comment_type", "issues")
        try:
            gh = type(self)._fn_get_github()
            gh.add_reaction(repo, comment_type, comment_id, "confused")
        except Exception:
            log.exception("failed to post error reaction on comment %s", comment_id)

    def _self_restart(self, repo_name: str, *, reason: str = "") -> None:
        runner_dir = type(self)._fn_runner_dir()
        self_repo = type(self)._fn_get_self_repo(runner_dir)
        if self_repo != repo_name:
            return  # Not our repo — nothing to do.
        log.info(
            "self-restart: %s on %s — syncing runner clone at %s",
            reason,
            repo_name,
            runner_dir,
        )
        # Sync runner BEFORE tearing down the worker.  If the sync fails we
        # log and return without touching the running workers — fido on the
        # kennel repo keeps running its old code rather than being silently
        # left without a worker thread.
        if not type(self)._fn_pull_with_backoff(runner_dir):
            log.error("self-restart: gave up — running old version (%s)", reason)
            return
        log.info(
            "self-restart: runner synced — stopping workers and re-execing (%s)", reason
        )
        self.registry.stop_and_join(repo_name)
        type(self)._fn_os_chdir(runner_dir)
        type(self)._fn_os_execvp("uv", ["uv", "run", "kennel", *sys.argv[1:]])

    def do_GET(self) -> None:
        if self.path == "/status":
            activities = []
            for a in self.registry.get_all_activities():
                crash = self.registry.get_crash_info(a.repo_name)
                activities.append(
                    {
                        "repo_name": a.repo_name,
                        "what": a.what,
                        "busy": a.busy,
                        "crash_count": crash.death_count if crash else 0,
                        "last_crash_error": crash.last_error if crash else None,
                        "is_stuck": self.registry.is_stale(
                            a.repo_name, _STALE_THRESHOLD
                        ),
                    }
                )
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


def _startup_pull(
    *,
    _runner_dir: Callable[[], Path] = _runner_dir,
    _get_head: Callable[..., str | None] = _get_head,
    _pull: Callable[..., bool] = _pull_with_backoff,
    _execvp: Callable[..., None] = os.execvp,
) -> None:
    """Sync the runner clone on startup and re-exec if HEAD changed."""
    runner_dir = _runner_dir()
    head_before = _get_head(runner_dir)
    if not _pull(runner_dir):
        log.warning("startup: runner sync failed — continuing with current code")
        return
    head_after = _get_head(runner_dir)
    if head_before and head_after and head_before != head_after:
        log.info(
            "startup: runner updated %s → %s — re-execing",
            head_before[:12],
            head_after[:12],
        )
        _execvp("uv", ["uv", "run", "kennel", *sys.argv[1:]])
    elif head_before and head_after:
        log.info("startup: runner already up to date at %s", head_before[:12])
    else:
        log.info("startup: runner synced (could not compare HEAD)")


def run(
    *,
    _from_args=Config.from_args,
    _HTTPServer=HTTPServer,
    _make_registry=make_registry,
    _path_home=Path.home,
    _basic_config=logging.basicConfig,
    _stderr=sys.stderr,
    _populate_memberships=populate_memberships,
    _signal=signal.signal,
    _kill_active_children=kill_active_children,
    _startup_pull=_startup_pull,
    _Watchdog=Watchdog,
    _preflight_repo_identity=preflight_repo_identity,
    _preflight_tools=preflight_tools,
    _preflight_sub_dir=preflight_sub_dir,
    _preflight_gh_auth=preflight_gh_auth,
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

    # Route uncaught exceptions through the logger so tracebacks land in
    # kennel.log, not just ~/log/kennel-crash.log (where start-kennel.sh
    # redirects stderr).  Before this, RCA on crashes required reading two
    # different log files.
    def _log_uncaught(exc_type, exc_value, exc_tb):
        log.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    def _log_thread_exception(args):
        log.critical(
            "uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_exception

    _startup_pull()
    _preflight_tools()
    _preflight_sub_dir(config)
    _preflight_gh_auth()
    _preflight_repo_identity(config.repos)

    _populate_memberships(config)

    WebhookHandler.config = config
    registry = _make_registry(config.repos)
    WebhookHandler.registry = registry
    _Watchdog(registry, config.repos).start_thread()

    server = _HTTPServer(("", config.port), WebhookHandler)

    def _shutdown_handler(signum: int, _frame: object) -> None:
        log.info("kennel received signal %d — terminating claude children", signum)
        _kill_active_children()
        server.server_close()
        sys.exit(0)

    _signal(signal.SIGTERM, _shutdown_handler)
    _signal(signal.SIGINT, _shutdown_handler)

    repos_str = ", ".join(f"{name}={rc.work_dir}" for name, rc in config.repos.items())
    log.info("kennel listening on :%d — repos: %s", config.port, repos_str)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        _kill_active_children()
        server.server_close()
