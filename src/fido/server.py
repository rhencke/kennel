import dataclasses
import faulthandler
import fcntl
import hashlib
import hmac
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from xml.etree.ElementTree import Element, SubElement, register_namespace, tostring

from fido import provider
from fido.claude import kill_active_children
from fido.config import Config, RepoConfig, RepoMembership
from fido.events import (
    Action,
    WebhookIngressOracle,
    create_task,
    dispatch,
    launch_worker,
    queue_reply_tasks,
    reply_outcome_creates_tasks,
    reply_to_comment,
    reply_to_issue_comment,
    reply_to_review,
)
from fido.github import GitHub
from fido.infra import (
    Clock,
    Filesystem,
    Infra,
    ProcessRunner,
    real_infra,
)
from fido.provider_factory import DefaultProviderFactory
from fido.rate_limit import RateLimitMonitor
from fido.registry import WebhookActivityHandle, WorkerRegistry, make_registry
from fido.rocq import self_restart as restart_fsm
from fido.state import State
from fido.static_files import StaticFiles
from fido.status import provider_statuses_for_repo_configs
from fido.store import FidoStore, ReplyPromiseRecord
from fido.tasks import Tasks, unblock_tasks
from fido.watchdog import (  # noqa: PLC2701
    _STALE_THRESHOLD,  # pyright: ignore[reportPrivateUsage]
    ReconcileWatchdog,
    Watchdog,
)
from fido.worker import RepoContextFilter

log = logging.getLogger(__name__)

# Exponential backoff for git pull during self-restart: 10s, 30s, 60s
# with a 10-minute total budget. Retries stop once the cumulative delay
# exceeds _PULL_BUDGET_SECONDS, even if a retry window remains.
_PULL_BACKOFF_DELAYS: tuple[int, ...] = (10, 30, 60)
_PULL_BUDGET_SECONDS: float = 600.0
_RESTART_EXIT_CODE = 75
_REQUEST_TIMEOUT_SECONDS = 10.0

# XML namespace URIs for the /status endpoint structural XML.
_NS_FIDO = "https://fidocancode.dog/fido"
_NS_DOG = "https://fidocancode.dog/woof"

# Register namespace prefixes for clean XML serialization.  Idempotent —
# safe to call at module scope before any threads start.
register_namespace("", _NS_FIDO)
register_namespace("dog", _NS_DOG)


class PreflightError(RuntimeError):
    """Raised by preflight checks when a startup precondition is not met.

    Caught by :func:`run` and converted to :exc:`SystemExit` so individual
    preflight functions remain testable without triggering process exit.
    """


class FidoHTTPServer(ThreadingHTTPServer):
    """Threaded webhook server with bounded per-connection reads.

    The standard ``HTTPServer`` handles one request at a time. A client that
    connects and stalls before sending a full HTTP request can otherwise block
    every webhook and status request behind it.
    """

    allow_reuse_address = True
    block_on_close = False
    daemon_threads = True
    request_queue_size = 64
    request_timeout_seconds = _REQUEST_TIMEOUT_SECONDS

    def get_request(self) -> tuple[Any, Any]:
        request, client_address = super().get_request()
        request.settimeout(self.request_timeout_seconds)
        return request, client_address


def _runner_dir() -> Path:
    """Return the runner clone directory — where the running fido code lives."""
    return Path(__file__).resolve().parents[1]


def _serialize_talker(talker: provider.SessionTalker | None) -> dict[str, Any] | None:
    """Convert a :class:`~fido.provider.SessionTalker` to a JSON-friendly dict.

    Returns ``None`` when nobody is talking to claude for the repo.
    """
    if talker is None:
        return None
    return {
        "repo_name": talker.repo_name,
        "thread_id": talker.thread_id,
        "kind": talker.kind,
        "description": talker.description,
        "claude_pid": talker.claude_pid,
        "started_at": talker.started_at.isoformat(),
    }


def _xml_text(value: object) -> str | None:
    """Convert a Python value to XML element text.

    Booleans become ``"true"`` / ``"false"``, ``None`` becomes ``None``
    (empty element), everything else becomes its ``str()`` representation.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _repo_status(act: dict[str, Any]) -> str:
    """Derive a status string from activity flags.

    Priority: paused > stuck > crashed > busy > what (falling back to
    "waiting").  Used as the ``dog:status`` attribute on ``<repo>`` elements
    so XSLT and CSS can style by state.  When the worker is not actively busy,
    the live ``what`` field (e.g. "waiting: no issues found") is used so the
    attribute carries the specific reason rather than a generic label.
    """
    provider_status = act.get("provider_status")
    if isinstance(provider_status, dict) and provider_status.get("paused"):
        return "paused"
    if act.get("is_stuck"):
        return "stuck"
    if act.get("crash_count", 0) > 0:
        return "crashed"
    if act.get("busy"):
        return "busy"
    return act.get("what") or "waiting"


def _activities_to_xml(payload: dict[str, Any]) -> bytes:
    """Serialize the full status payload to namespaced structural XML with an XSLT PI.

    Pure function — transforms data, no I/O.  The server emits this structural
    XML in the ``https://fidocancode.dog/fido`` namespace.  The browser
    fetches ``status.xsl`` (via the XSLT processing instruction), which
    transforms it into display-oriented XML in a separate namespace, which
    is then styled by ``status.css`` via a CSS processing instruction.

    Three-layer pipeline: structural XML → XSLT → display XML → CSS.

    The root ``<fido>`` element carries fido-level metadata (uptime, rate
    limits) before the per-repo ``<repo>`` children so XSLT can render a
    dashboard header without reaching into any individual repo card.
    """
    root = Element(f"{{{_NS_FIDO}}}fido")

    # Fido-level metadata — emitted before per-repo elements.
    fido_uptime = payload.get("fido_uptime_seconds")
    if fido_uptime is not None:
        el = SubElement(root, f"{{{_NS_FIDO}}}fido_uptime_seconds")
        el.text = _xml_text(fido_uptime)

    rate_limit = payload.get("rate_limit")
    if rate_limit is not None:
        rl_el = SubElement(root, f"{{{_NS_FIDO}}}rate_limit")
        for rl_key, rl_val in rate_limit.items():
            if isinstance(rl_val, dict):
                child_el = SubElement(rl_el, f"{{{_NS_FIDO}}}{rl_key}")
                for k, v in rl_val.items():
                    sub_el = SubElement(child_el, f"{{{_NS_FIDO}}}{k}")
                    sub_el.text = _xml_text(v)
            else:
                child_el = SubElement(rl_el, f"{{{_NS_FIDO}}}{rl_key}")
                child_el.text = _xml_text(rl_val)

    for act in payload.get("activities", []):
        repo = SubElement(root, f"{{{_NS_FIDO}}}repo")
        repo.set(f"{{{_NS_DOG}}}status", _repo_status(act))
        for key, value in act.items():
            if key == "webhook_activities":
                wa_el = SubElement(repo, f"{{{_NS_FIDO}}}webhook_activities")
                for wh in value:
                    webhook_el = SubElement(wa_el, f"{{{_NS_FIDO}}}webhook")
                    for wk, wv in wh.items():
                        el = SubElement(webhook_el, f"{{{_NS_FIDO}}}{wk}")
                        el.text = _xml_text(wv)
            elif key == "claude_talker":
                ct_el = SubElement(repo, f"{{{_NS_FIDO}}}claude_talker")
                if value is not None:
                    for ck, cv in value.items():
                        el = SubElement(ct_el, f"{{{_NS_FIDO}}}{ck}")
                        el.text = _xml_text(cv)
            elif key == "provider_status":
                ps_el = SubElement(repo, f"{{{_NS_FIDO}}}provider_status")
                if value is not None:
                    for pk, pv in value.items():
                        el = SubElement(ps_el, f"{{{_NS_FIDO}}}{pk}")
                        el.text = _xml_text(pv)
            elif key == "issue_cache":
                ic_el = SubElement(repo, f"{{{_NS_FIDO}}}issue_cache")
                if value is not None:
                    for ik, iv in value.items():
                        el = SubElement(ic_el, f"{{{_NS_FIDO}}}{ik}")
                        el.text = _xml_text(iv)
            else:
                el = SubElement(repo, f"{{{_NS_FIDO}}}{key}")
                el.text = _xml_text(value)
    xml_body = tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<?xml-stylesheet type="text/xsl" href="/static/status.xsl"?>\n' + xml_body
    ).encode("utf-8")


def _parse_repo_from_url(url: str) -> str | None:
    """Extract 'owner/repo' from an SSH or HTTPS git remote URL, or return None."""
    parsed = urlparse(url)
    if parsed.scheme:
        # Standard URL (https, ssh, git, etc.): path is /owner/repo[.git]
        path = parsed.path
    elif ":" in url:
        # SCP-style SSH: git@github.com:owner/repo[.git]
        _, path = url.split(":", 1)
    else:
        return None
    path = path.lstrip("/").removesuffix(".git")
    parts = path.split("/")
    return path if len(parts) == 2 and all(parts) else None


def _serialize_provider_status(status: Any) -> dict[str, Any] | None:
    """Convert a ProviderPressureStatus to a JSON-friendly dict."""
    if status is None:
        return None
    return {
        "provider": status.provider,
        "window_name": status.window_name,
        "pressure": status.pressure,
        "percent_used": status.percent_used,
        "resets_at": status.resets_at.isoformat() if status.resets_at else None,
        "unavailable_reason": status.unavailable_reason,
        "level": status.level,
        "warning": status.warning,
        "paused": status.paused,
    }


def _serialize_rate_limit(monitor: Any) -> dict[str, Any] | None:
    """Serialize the latest :class:`~fido.rate_limit.RateLimitSnapshot`
    for the ``/status.json`` payload (closes #812 follow-up).

    Returns ``None`` when *monitor* is missing (tests with a MagicMock
    registry omit it) or hasn't yet completed its first refresh.
    """
    from fido.rate_limit import RateLimitMonitor

    if not isinstance(monitor, RateLimitMonitor):
        return None
    snap = monitor.latest()
    if snap is None:
        return None
    return {
        "rest": _serialize_rate_window(snap.rest),
        "graphql": _serialize_rate_window(snap.graphql),
        "fetched_at": snap.fetched_at.isoformat(),
    }


def _serialize_rate_window(window: Any) -> dict[str, Any]:
    return {
        "name": window.name,
        "used": window.used,
        "limit": window.limit,
        "resets_at": window.resets_at.isoformat(),
    }


def _serialize_issue_cache(cache: Any) -> dict[str, Any] | None:
    """Serialize an :class:`~fido.issue_cache.IssueTreeCache` snapshot for
    the /status.json payload (#812).  Returns ``None`` when the cache has
    not been bootstrapped yet (or when *cache* isn't a real cache, which
    happens in tests that hand the server a MagicMock registry) — fido
    status hides the line in that case.
    """
    from fido.issue_cache import IssueTreeCache

    if not isinstance(cache, IssueTreeCache):
        return None
    metrics = cache.metrics()
    return {
        "loaded": metrics.inventory_loaded_at is not None,
        "open_issues": metrics.open_issue_count,
        "events_applied": metrics.events_applied,
        "events_dropped_stale": metrics.events_dropped_stale,
        "last_event_at": metrics.last_event_at.isoformat()
        if metrics.last_event_at
        else None,
        "last_reconcile_at": metrics.last_reconcile_at.isoformat()
        if metrics.last_reconcile_at
        else None,
        "last_reconcile_drift": metrics.last_reconcile_drift,
    }


def _collect_fido_state(work_dir: Path, now: datetime) -> dict[str, Any]:
    """Read fido filesystem state (issue, PR, tasks, lock) for the status endpoint.

    Best-effort: returns safe defaults on any I/O or parse error so that a
    missing or partially-written state file never breaks the status endpoint.
    """
    fido_dir = work_dir / ".git" / "fido"

    # Check whether the fido sub-process lock is held.
    fido_running = False
    lock_path = fido_dir / "lock"
    if lock_path.exists():
        try:
            fd = open(lock_path)  # noqa: SIM115
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                fido_running = True
            finally:
                fd.close()
        except OSError:
            pass

    # Load state.json (issue / PR metadata).
    try:
        state = State(fido_dir).load()
    except Exception:  # noqa: BLE001
        log.warning("_collect_fido_state: failed to load state for %s", work_dir)
        state = {}
    issue: int | None = state.get("issue")
    issue_title: str | None = state.get("issue_title")
    issue_elapsed_seconds: int | None = None
    issue_started_at: str | None = state.get("issue_started_at")
    if issue_started_at:
        try:
            started = datetime.fromisoformat(issue_started_at)
            issue_elapsed_seconds = max(0, int((now - started).total_seconds()))
        except TypeError, ValueError:
            pass
    pr_number: int | None = state.get("pr_number")
    pr_title: str | None = state.get("pr_title")

    # Load tasks.json.
    try:
        task_list = Tasks(work_dir).list()
    except Exception:  # noqa: BLE001
        log.warning("_collect_fido_state: failed to load tasks for %s", work_dir)
        task_list = []
    pending = sum(1 for t in task_list if t.get("status") == "pending")
    completed = sum(1 for t in task_list if t.get("status") == "completed")

    current_task: str | None = None
    for t in task_list:
        if t.get("status") == "in_progress":
            current_task = t.get("title")
            break
    if current_task is None:
        for t in task_list:
            if t.get("status") == "pending":
                current_task = t.get("title")
                break

    non_completed = [t for t in task_list if t.get("status") != "completed"]
    task_number: int | None = None
    task_total: int | None = None
    if non_completed:
        task_total = len(non_completed)
        for idx, t in enumerate(non_completed, start=1):
            if t.get("status") == "in_progress":
                task_number = idx
                break
        if task_number is None:
            task_number = 1

    return {
        "fido_running": fido_running,
        "issue": issue,
        "issue_title": issue_title,
        "issue_elapsed_seconds": issue_elapsed_seconds,
        "pr_number": pr_number,
        "pr_title": pr_title,
        "pending": pending,
        "completed": completed,
        "current_task": current_task,
        "task_number": task_number,
        "task_total": task_total,
    }


def _get_self_repo(runner_dir: Path, proc: ProcessRunner) -> str | None:
    """Return 'owner/repo' from the runner clone's origin remote, or None on error.

    Handles both SSH (``git@github.com:owner/repo.git``) and HTTPS
    (``https://github.com/owner/repo.git``) remote URLs.
    """
    try:
        result = proc.run(
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
    proc: ProcessRunner,
) -> None:
    """Verify each configured work_dir is a git repo whose origin matches its name.

    Raises :exc:`PreflightError` if any repo's origin remote can't be read,
    can't be parsed, or doesn't match the configured ``owner/repo`` name.
    """
    for name, repo_cfg in repos.items():
        try:
            result = proc.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repo_cfg.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise PreflightError(
                f"preflight: {name}: git remote get-url failed: {e}"
            ) from e
        except FileNotFoundError as e:
            raise PreflightError(f"preflight: {name}: git not found: {e}") from e
        url = result.stdout.strip()
        actual = _parse_repo_from_url(url)
        if actual is None:
            raise PreflightError(
                f"preflight: {name}: could not parse owner/repo from origin remote: {url!r}"
            )
        if actual != name:
            raise PreflightError(
                f"preflight: {name}: origin remote is {actual!r} — expected {name!r}"
            )
        log.info("preflight: %s: work_dir identity confirmed", name)


_REQUIRED_TOOLS = ("git", "gh", "claude", "copilot", "codex")


def preflight_tools(fs: Filesystem) -> None:
    """Verify that all required CLI tools are on PATH.

    Raises :exc:`PreflightError` naming the first missing tool.
    """
    for tool in _REQUIRED_TOOLS:
        if fs.which(tool) is None:
            raise PreflightError(
                f"preflight: required tool not found on PATH: {tool!r}"
            )
    log.info("preflight: all required tools found: %s", ", ".join(_REQUIRED_TOOLS))


def preflight_sub_dir(config: Config, fs: Filesystem) -> None:
    """Verify that the skill-files directory exists.

    Raises :exc:`PreflightError` if ``config.sub_dir`` is not an existing
    directory.  Workers read ``persona.md`` and sub-skill files from here on
    every task run — a missing directory causes every worker invocation to fail
    with an obscure I/O error rather than a clear startup message.
    """
    if not fs.is_dir(config.sub_dir):
        raise PreflightError(
            f"preflight: skill-files directory not found: {config.sub_dir}"
        )
    log.info("preflight: skill-files directory confirmed: %s", config.sub_dir)


def preflight_gh_auth(gh: GitHub) -> None:
    """Verify gh auth works by fetching the authenticated bot user.

    Raises :exc:`PreflightError` if ``get_user()`` fails for any reason
    (bad token, network error, etc.).
    """
    try:
        bot_user = gh.get_user()
    except Exception as e:
        raise PreflightError(f"preflight: gh auth check failed: {e}") from e
    log.info("preflight: gh auth confirmed — bot user is %r", bot_user)


def _pull_with_backoff(
    runner_dir: Path,
    proc: ProcessRunner,
    clock: Clock,
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
    start = clock.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            proc.run(
                ["git", "fetch", "origin", "main"],
                cwd=str(runner_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            proc.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=str(runner_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            log.info(
                "self-restart: runner synced on attempt %d (%.1fs elapsed)",
                attempt,
                clock.monotonic() - start,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            elapsed = clock.monotonic() - start
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
            clock.sleep(delay)


def _spawn_bg(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    """Spawn *fn* in a background daemon thread."""
    threading.Thread(target=fn, args=args, daemon=True).start()


def _noop_after_post() -> None:
    """Default no-op hook called at the end of do_POST.

    Tests override _fn_after_do_post to synchronise without sleeping — the
    hook fires after _fn_spawn_bg so any captured background thread is in
    the capture list before the test wakes up.
    """


class WebhookHandler(BaseHTTPRequestHandler):
    config: Config
    registry: WorkerRegistry
    provider_factory: DefaultProviderFactory | None = None
    rate_limit_monitor: Any = None
    # Set by run() to record when the server came up, used for fido uptime.
    fido_started_at: datetime | None = None
    # Injectable collaborators — set as class attributes so HTTP-driven tests
    # can replace them without patching module-level names.
    gh: GitHub | None = None
    # Infrastructure ports — set by server.run() composition root.
    infra: Infra = real_infra()
    static_files: StaticFiles | None = None
    _fn_dispatch = staticmethod(dispatch)
    _fn_reply_to_comment = staticmethod(reply_to_comment)
    _fn_reply_to_review = staticmethod(reply_to_review)
    _fn_reply_to_issue_comment = staticmethod(reply_to_issue_comment)
    _fn_create_task = staticmethod(create_task)
    _fn_launch_worker = staticmethod(launch_worker)
    _fn_unblock_tasks = staticmethod(unblock_tasks)
    _fn_spawn_bg = staticmethod(_spawn_bg)
    _fn_after_do_post = staticmethod(_noop_after_post)
    _fn_runner_dir = staticmethod(_runner_dir)
    _fn_kill_active_children = staticmethod(kill_active_children)
    # Per-process ingress deduplication oracle (webhook_ingress_dedupe.v).
    # Shared across all handler instances via the class attribute so every
    # request on every thread sees the same delivery-ID table.  Created once
    # at class definition time — no lock needed here because the attribute is
    # set before any threads start and never replaced.
    ingress_oracle: WebhookIngressOracle = WebhookIngressOracle()
    # Process-level FSM state from self_restart.v.  Tracks the restart episode
    # in progress so coordination violations surface as immediate crashes.
    # Initialised to Running() at class definition — the process is always
    # "running normally" before any restart trigger fires.  Reset to Running()
    # after an Aborted episode so a subsequent trigger can begin a fresh one.
    _restart_fsm_state: restart_fsm.State = restart_fsm.Running()

    def _restart_fsm_transition(self, event: restart_fsm.Event) -> restart_fsm.State:
        """Fire *event* against the process-level self-restart FSM.

        Raises :exc:`AssertionError` if the transition is rejected — a
        restart-sequence violation crashes loudly rather than silently
        continuing in an undefined state.

        Uses ``type(self)`` so the class-level state is updated and visible
        to every subsequent handler instance in the same process.
        """
        prev = type(self)._restart_fsm_state
        new_state = restart_fsm.transition(prev, event)
        if new_state is None:
            raise AssertionError(
                f"self_restart FSM: {type(event).__name__} rejected in "
                f"state {type(prev).__name__}"
            )
        type(self)._restart_fsm_state = new_state
        log.debug(
            "self-restart FSM: %s →%s via %s",
            type(prev).__name__,
            type(new_state).__name__,
            type(event).__name__,
        )
        return new_state

    def do_POST(self) -> None:
        try:
            self._do_post_inner()
        finally:
            type(self)._fn_after_do_post()

    def _do_post_inner(self) -> None:
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
            action = type(self)._fn_dispatch(
                event,
                payload,
                self.config,
                repo_cfg,
                delivery_id=delivery,
                oracle=type(self).ingress_oracle,
            )
        except Exception:
            log.exception("dispatch failed for %s", repo_name)
            self._respond(500, "dispatch error")
            return

        # Patch the issue tree cache before ACK too — failure here is
        # recoverable (hourly reconcile heals drift), so log and continue
        # rather than 500-ing the webhook (#812).
        try:
            self._patch_issue_cache(event, payload, repo_cfg)
        except Exception:
            log.exception(
                "issue-cache patch failed for %s — hourly reconcile will heal",
                repo_name,
            )

        # Acknowledge only after dispatch succeeds.
        self._respond(200, "ok")

        # Self-restart after ack so the response reaches GitHub before exec.
        if is_pr_merged:
            self._self_restart(repo_name, reason="PR merged")
        elif is_default_push:
            self._self_restart(repo_name, reason=f"push to {default_branch}")

        # Fire the preemption signal synchronously on the HTTP handler thread
        # BEFORE spawning the background thread (#955).  If a worker turn is
        # in flight, this cancels it immediately — eliminating the race where
        # the background thread might be delayed past the end of the turn and
        # arrive too late to preempt.  The background thread's hold_for_handler
        # still fires a second preempt in case the worker starts a new turn in
        # the window between this cancel and the background thread acquiring
        # the session lock.
        #
        # Also enter the untriaged inbox (#1067) synchronously here — before
        # the background thread spawns — so the worker sees a non-empty inbox
        # at its next turn boundary and yields rather than starting another
        # provider turn.  exit_untriaged is called in _process_action's finally
        # block when the handler finishes.
        if action and self._action_preempts_worker(action):
            session = self.registry.get_session(repo_cfg.name)
            if session is not None:
                session.preempt_worker()
            self.registry.enter_untriaged(repo_cfg.name)

        # Process in background thread so we don't block the server.
        if action:
            type(self)._fn_spawn_bg(self._process_action, (action, repo_cfg))

    def _patch_issue_cache(
        self, event: str, payload: dict[str, Any], repo_cfg: RepoConfig
    ) -> None:
        """Translate the webhook to a cache event and apply it to the
        per-repo :class:`~fido.issue_cache.IssueTreeCache` (#812).

        No-op when the event isn't picker-relevant (translator returns
        ``None``).  No-op also when the cache hasn't been initialized —
        the worker thread bootstraps the cache on its first iteration,
        and any events arriving before that get queued by the cache
        itself and drained when the inventory load completes.
        """
        from fido.cache_webhooks import translate

        translation = translate(event, payload)
        if translation is None:
            log.debug(
                "issue-cache[%s]: webhook %s/%s not picker-relevant — skipping",
                repo_cfg.name,
                event,
                payload.get("action", "?"),
            )
            return
        cache_event_type, cache_payload = translation
        log.info(
            "issue-cache[%s]: applying %s for #%s (from webhook %s/%s)",
            repo_cfg.name,
            cache_event_type,
            cache_payload.get("issue_number"),
            event,
            payload.get("action", "?"),
        )
        cache = self.registry.get_issue_cache(repo_cfg.name)
        cache.apply_event(cache_event_type, cache_payload)

    def _process_action(self, action: Action, repo_cfg: RepoConfig) -> None:
        description = self._describe_action(action)
        tid = threading.get_ident()
        log.info(
            "webhook handler: ENTER repo=%s description=%r tid=%d",
            repo_cfg.name,
            description,
            tid,
        )
        provider.set_thread_repo(repo_cfg.name)
        provider.set_thread_kind("webhook")
        session = self.registry.get_session(repo_cfg.name)
        needs_model = self._action_uses_model(action)
        preempts_worker = self._action_preempts_worker(action)
        try:
            with self.registry.webhook_activity(repo_cfg.name, description) as activity:
                if session is not None and needs_model:
                    # Hold the session across the whole handler (#658) so the
                    # worker can't sneak in and acquire the lock between this
                    # handler's individual turns — that stalled webhook replies
                    # behind long worker turns.  Both ClaudeSession and
                    # CopilotCLISession implement ``hold_for_handler``.
                    with session.hold_for_handler(preempt_worker=True):
                        self._process_action_inner(action, repo_cfg, activity)
                else:
                    self._process_action_inner(action, repo_cfg, activity)
        except provider.SessionLeakError:
            # A webhook and a worker tried to talk to the same repo's claude
            # at the same time — the only safe action is to halt the whole
            # process so the supervisor (start-fido.sh) restarts us fresh.
            log.exception(
                "claude leak detected in webhook handler for %s — halting",
                repo_cfg.name,
            )
            os._exit(3)
        finally:
            log.info(
                "webhook handler: EXIT repo=%s tid=%d",
                repo_cfg.name,
                tid,
            )
            if preempts_worker:
                # Mirror the enter_untriaged called synchronously in
                # _do_post_inner — decrement now that this handler action is
                # done so the worker can resume its next turn (#1067).  Model
                # actions hold the provider session; non-model interrupt
                # actions such as CI failures still block worker admission
                # until the webhook action has launched/woken the worker.
                self.registry.exit_untriaged(repo_cfg.name)
            provider.set_thread_kind(None)
            provider.set_thread_repo(None)

    def _action_uses_model(self, action: Action) -> bool:
        """True when the webhook action will call ``agent.run_turn``.

        Reply-capable actions (PR comments, review comments, review threads)
        generate a response through the model and therefore benefit from
        holding the session lock across their whole handler (#658).  Plain
        webhook-action events (merges, check_runs) only restart workers and
        don't touch the model — no point blocking the worker for those.
        """
        return bool(action.reply_to or action.review_comments or action.comment_body)

    def _action_preempts_worker(self, action: Action) -> bool:
        """True when the action must run before the next worker provider turn."""
        return self._action_uses_model(action) or action.preempts_worker

    def _describe_action(self, action: Action) -> str:
        """Short label for status display — what this webhook handler is doing."""
        if action.reply_to:
            return "handling review comment"
        if action.review_comments:
            return "handling review thread"
        if action.comment_body:
            return "handling PR comment"
        return "handling webhook action"

    def _reply_promise(self, action: Action) -> tuple[str, int] | None:
        """Return the durable reply-promise key for reply-capable webhook actions."""
        thread = action.reply_to or action.thread
        if not thread:
            return None
        comment_type = thread["comment_type"]
        comment_id = thread["comment_id"]
        if comment_type not in {"issues", "pulls"}:
            raise ValueError(f"invalid reply promise comment type: {comment_type!r}")
        if not isinstance(comment_id, int):
            raise TypeError(f"invalid reply promise comment id: {comment_id!r}")
        return comment_type, comment_id

    def _prepare_reply(
        self,
        repo_cfg: RepoConfig,
        action: Action,
    ) -> ReplyPromiseRecord | None:
        """Claim an action's raw comment id and attach its promise marker."""
        promise_key = self._reply_promise(action)
        if promise_key is None:
            return None
        comment_type, comment_id = promise_key
        promise = FidoStore(repo_cfg.work_dir).prepare_reply(
            owner="webhook",
            comment_type=comment_type,
            anchor_comment_id=comment_id,
        )
        if promise is None:
            log.info("already replied to comment %s — skipping", comment_id)
            return None
        action.context = {
            **(action.context or {}),
            "reply_promise_id": promise.promise_id,
        }
        return promise

    def _ack_reply(
        self, repo_cfg: RepoConfig, promise: ReplyPromiseRecord | None
    ) -> None:
        """Mark a reply promise completed after its handler returns."""
        if promise is None:
            return
        store = FidoStore(repo_cfg.work_dir)
        store.mark_posted(promise.promise_id)
        store.ack_promise(promise.promise_id)

    def _fail_reply(
        self, repo_cfg: RepoConfig, promise: ReplyPromiseRecord | None
    ) -> None:
        """Mark a reply promise retryable after a handler failure."""
        if promise is not None:
            FidoStore(repo_cfg.work_dir).mark_failed(promise.promise_id)

    def _process_action_inner(
        self,
        action: Action,
        repo_cfg: RepoConfig,
        activity: WebhookActivityHandle,
    ) -> None:
        # The worker thread's own ``worker_what`` is not touched here — this
        # handler runs on a separate webhook thread and its activity is
        # surfaced in the repo's :class:`~fido.registry.WebhookActivity`
        # list (via :meth:`~fido.registry.WorkerRegistry.webhook_activity`).
        # Writing here would clobber the worker thread's own state, which is
        # what caused the old ``Doing: handling webhook action`` display bug.
        try:
            gh = cast(GitHub, self.gh)  # always set by serve() before first request
            handled = False
            category: str | None = None
            titles: list[str] = []
            queued_tasks = 0

            if action.reply_to:
                promise = self._prepare_reply(repo_cfg, action)
                if promise is None:
                    handled = True
                    category, titles = None, []
                else:
                    activity.set_description("triaging review comment")
                    try:
                        category, titles = type(self)._fn_reply_to_comment(
                            action, self.config, repo_cfg, gh
                        )
                    except Exception:
                        self._fail_reply(repo_cfg, promise)
                        raise
                    self._ack_reply(repo_cfg, promise)
                    handled = True
                # Create task based on triage result.
                # DEFER files a GitHub issue (handled in reply_to_comment) — no tasks.json entry.
                # ACT, DO → add each task title to work queue.
                if category is not None:
                    if reply_outcome_creates_tasks(
                        category,
                        thread=action.reply_to,
                        is_bot=action.is_bot,
                    ):
                        activity.set_description(
                            "queuing review comment tasks"
                            if len(titles or []) != 1
                            else "queuing review comment task"
                        )
                    queued_tasks += queue_reply_tasks(
                        category,
                        titles or [],
                        self.config,
                        repo_cfg,
                        gh,
                        thread=action.reply_to,
                        is_bot=action.is_bot,
                        registry=self.registry,
                        create_task_fn=type(self)._fn_create_task,
                    )

            if action.review_comments:
                activity.set_description("replying to review thread")
                type(self)._fn_reply_to_review(action, self.config, repo_cfg, gh)
                handled = True  # inline comments handled individually

            # Top-level PR comments (issue_comment) — no reply_to, but has comment_body
            if not handled and action.comment_body:
                promise = self._prepare_reply(repo_cfg, action)
                if promise is None:
                    category, titles = None, []
                else:
                    activity.set_description("triaging PR comment")
                    try:
                        category, titles = type(self)._fn_reply_to_issue_comment(
                            action, self.config, repo_cfg, gh
                        )
                    except Exception:
                        self._fail_reply(repo_cfg, promise)
                        raise
                    self._ack_reply(repo_cfg, promise)
                handled = True
                # DEFER files a GitHub issue — no tasks.json entry.
                if reply_outcome_creates_tasks(
                    category or "",
                    thread=action.thread,
                    is_bot=action.is_bot,
                ):
                    activity.set_description(
                        "queuing PR comment tasks"
                        if len(titles) != 1
                        else "queuing PR comment task"
                    )
                queued_tasks += queue_reply_tasks(
                    category or "",
                    titles,
                    self.config,
                    repo_cfg,
                    gh,
                    thread=action.thread,
                    is_bot=action.is_bot,
                    registry=self.registry,
                    create_task_fn=type(self)._fn_create_task,
                )

            log.info(
                "action outcome: handled=%s category=%s tasks=%d",
                handled,
                category,
                queued_tasks,
            )
            # When a human comments on a PR, transition any BLOCKED tasks back
            # to PENDING so the worker can re-evaluate and resume.
            if action.reply_to or action.comment_body:
                type(self)._fn_unblock_tasks(repo_cfg.work_dir)
            # Non-comment events just trigger fido worker — no task needed
            type(self)._fn_launch_worker(repo_cfg, self.registry)
        except provider.SessionLeakError:
            # Let the outer _process_action handler halt fido — we must not
            # swallow a leak into the generic "confused reaction" path below.
            raise
        except Exception:
            log.exception("error processing action")
            self._signal_action_error(action)

    def _signal_action_error(self, action: Action) -> None:
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
        if not repo or not comment_id:
            return
        try:
            if self.gh is not None:
                self.gh.add_reaction(repo, comment_type, comment_id, "confused")
        except Exception:
            log.exception("failed to post error reaction on comment %s", comment_id)

    def _self_restart(self, repo_name: str, *, reason: str = "") -> None:
        runner_dir = type(self)._fn_runner_dir()
        self_repo = _get_self_repo(runner_dir, self.infra.proc)
        if self_repo != repo_name:
            return  # Not our repo — nothing to do.
        # FSM oracle: Running → Syncing.  This fires before any side effects so
        # a double-trigger (two webhooks racing) raises AssertionError on the
        # second call rather than tearing down workers a second time.
        self._restart_fsm_transition(restart_fsm.TriggerRestart())
        log.info(
            "self-restart: %s on %s — syncing runner clone at %s",
            reason,
            repo_name,
            runner_dir,
        )
        # Sync runner BEFORE tearing down the worker.  If the sync fails we
        # log and return without touching the running workers — fido on the
        # fido repo keeps running its old code rather than being silently
        # left without a worker thread.
        if not _pull_with_backoff(runner_dir, self.infra.proc, self.infra.clock):
            log.error("self-restart: gave up — running old version (%s)", reason)
            # FSM oracle: Syncing → Aborted.  Validates sync_before_teardown:
            # the FSM reaches Aborted without passing through StoppingWorkers,
            # confirming workers were never touched.  Reset to Running() so a
            # subsequent trigger can begin a fresh episode.
            self._restart_fsm_transition(restart_fsm.SyncFail())
            type(self)._restart_fsm_state = restart_fsm.Running()
            return
        # FSM oracle: Syncing → StoppingWorkers.
        self._restart_fsm_transition(restart_fsm.SyncOk())
        log.info(
            "self-restart: runner synced — stopping workers and exiting %d (%s)",
            _RESTART_EXIT_CODE,
            reason,
        )
        # Stop the merged repo's worker cleanly.
        self.registry.stop_and_join(repo_name)
        # Tear down every remaining worker and SIGTERM all tracked claude
        # subprocesses before asking the host supervisor to restart (closes
        # #829). Any subprocess not explicitly killed here gets reparented to
        # init and keeps running after restart: accepting its still-open stdin,
        # still writing to the workspace.
        # We've seen this orphan a session that then committed + reset
        # over an in-progress human edit hours after fido was "stopped".
        self.registry.stop_all()
        # FSM oracle: StoppingWorkers → KillingChildren.  Validates
        # workers_before_children: all workers stopped before kill fires.
        self._restart_fsm_transition(restart_fsm.WorkersStopped())
        type(self)._fn_kill_active_children()
        # FSM oracle: KillingChildren → Exiting.  Validates
        # exit_requires_full_teardown: process only exits after full teardown.
        self._restart_fsm_transition(restart_fsm.ChildrenKilled())
        self.infra.os_proc.chdir(runner_dir)
        self.infra.os_proc.exit(_RESTART_EXIT_CODE)

    def do_GET(self) -> None:
        if self.path == "/status":
            body = _activities_to_xml(self._collect_status_payload())
            self._respond_body("application/xml; charset=utf-8", body)
        elif self.path == "/status.json":
            body = json.dumps(self._collect_status_payload()).encode()
            self._respond_body("application/json", body)
        elif self.path.startswith("/static/"):
            self._serve_static()
        else:
            self._respond(200, "fido is running")

    def _collect_status_payload(self) -> dict[str, Any]:
        """Build the status payload shared by ``/status.json`` and ``/status`` XML.

        Wraps the per-repo activity list, a global rate-limit snapshot, and
        fido-level metadata (uptime) under top-level keys.  Both endpoints
        call this method so they always render the same data.
        """
        now = datetime.now(tz=timezone.utc)
        started = self.fido_started_at
        fido_uptime = (now - started).total_seconds() if started is not None else None
        return {
            "activities": self._collect_activities(),
            "rate_limit": _serialize_rate_limit(self.rate_limit_monitor),
            "fido_uptime_seconds": fido_uptime,
        }

    def _collect_activities(self) -> list[dict[str, Any]]:
        """Build the activity snapshot for all registered repos."""
        now = datetime.now(tz=timezone.utc)
        activities: list[dict[str, Any]] = []
        provider_statuses = provider_statuses_for_repo_configs(
            list(self.config.repos.values()),
            _provider_factory=self.provider_factory,
        )
        for a in self.registry.get_all_activities():
            crash = self.registry.get_crash_info(a.repo_name)
            started_at = self.registry.thread_started_at(a.repo_name)
            repo_cfg = self.config.repos.get(a.repo_name)
            dropped_count = int(self.registry.get_session_dropped_count(a.repo_name))
            worker_uptime = (
                (now - started_at).total_seconds() if started_at is not None else None
            )
            webhooks = [
                {
                    "description": w.description,
                    "elapsed_seconds": (now - w.started_at).total_seconds(),
                    "thread_id": w.thread_id,
                }
                for w in self.registry.get_webhook_activities(a.repo_name)
            ]
            fido_state = (
                _collect_fido_state(repo_cfg.work_dir, now)
                if repo_cfg is not None
                else {
                    "fido_running": False,
                    "issue": None,
                    "issue_title": None,
                    "issue_elapsed_seconds": None,
                    "pr_number": None,
                    "pr_title": None,
                    "pending": 0,
                    "completed": 0,
                    "current_task": None,
                    "task_number": None,
                    "task_total": None,
                }
            )
            activities.append(
                {
                    "repo_name": a.repo_name,
                    "what": a.what,
                    "busy": a.busy,
                    "crash_count": crash.death_count if crash else 0,
                    "last_crash_error": crash.last_error if crash else None,
                    "is_stuck": self.registry.is_stale(a.repo_name, _STALE_THRESHOLD),
                    "worker_uptime_seconds": worker_uptime,
                    "webhook_activities": webhooks,
                    "provider": repo_cfg.provider if repo_cfg is not None else None,
                    "provider_status": _serialize_provider_status(
                        provider_statuses.get(repo_cfg.provider)
                        if repo_cfg is not None
                        else None
                    ),
                    "session_owner": self.registry.get_session_owner(a.repo_name),
                    "session_alive": self.registry.get_session_alive(a.repo_name),
                    "session_pid": self.registry.get_session_pid(a.repo_name),
                    "session_dropped_count": dropped_count,
                    "session_sent_count": int(
                        self.registry.get_session_sent_count(a.repo_name)
                    ),
                    "session_received_count": int(
                        self.registry.get_session_received_count(a.repo_name)
                    ),
                    "claude_talker": _serialize_talker(
                        provider.get_talker(a.repo_name)
                    ),
                    "rescoping": self.registry.is_rescoping(a.repo_name),
                    "issue_cache": _serialize_issue_cache(
                        self.registry.get_issue_cache(a.repo_name)
                    ),
                    **fido_state,
                }
            )
        return activities

    def _respond_body(self, content_type: str, body: bytes) -> None:
        """Send a 200 response with the given content type and body."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self) -> None:
        if self.static_files is None:
            self._respond(404, "not found")
            return
        response = self.static_files.serve(
            self.path,
            self.headers.get("If-None-Match"),
            self.headers.get("If-Modified-Since"),
        )
        if response is None:
            self._respond(404, "not found")
            return
        self.send_response(response.status)
        for name, value in response.headers:
            self.send_header(name, value)
        self.end_headers()
        if response.body:
            self.wfile.write(response.body)

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


def populate_memberships(config: Config, gh: GitHub) -> None:
    """Fetch collaborators for each repo once at startup and store on RepoConfig.

    Mutates ``config.repos`` in place — each :class:`RepoConfig` is replaced
    with a new instance carrying a populated :class:`RepoMembership`.  Uses
    the provided GitHub client instance for all repos.  Bot account (gh_user)
    is excluded from every collaborator set.
    """
    bot_user = gh.get_user()
    for name, repo_cfg in list(config.repos.items()):
        collabs = frozenset(c for c in gh.get_collaborators(name) if c != bot_user)
        log.info("%s: collaborators = %s", name, sorted(collabs) or "(none)")
        config.repos[name] = dataclasses.replace(
            repo_cfg, membership=RepoMembership(collaborators=collabs)
        )


def bootstrap_issue_caches(
    repos: dict[str, RepoConfig],
    gh: GitHub,
    registry: WorkerRegistry,
) -> None:
    """Bootstrap every per-repo :class:`~fido.issue_cache.IssueTreeCache` at startup (#837).

    Called once in :func:`run` after the registry is created but before the
    watchdog threads start.  Each repo gets a fresh ``find_all_open_issues``
    snapshot so the cache is populated from the first moment — even for repos
    whose worker resumes on an existing issue and never calls
    ``find_next_issue`` during the current fido run.

    After each successful load, the worker thread is woken so it rescans
    immediately rather than waiting out the 60-second idle timeout (#995).

    Per-repo failures are swallowed (logged, not raised): a single GitHub
    API hiccup must not prevent fido from starting.  The hourly
    :class:`~fido.watchdog.ReconcileWatchdog` will heal any cold repo
    within the hour.
    """
    for name in repos:
        owner, repo_name = name.split("/", 1)
        cache = registry.get_issue_cache(name)
        try:
            snapshot_started_at = datetime.now(tz=timezone.utc)
            log.info("startup: bootstrapping issue cache for %s", name)
            inventory = gh.find_all_open_issues(owner, repo_name)
            cache.load_inventory(inventory, snapshot_started_at=snapshot_started_at)
            registry.wake(name)
        except Exception:
            log.exception(
                "startup: failed to bootstrap issue cache for %s — "
                "ReconcileWatchdog will heal within the hour",
                name,
            )


def run(
    *,
    _from_args: Callable[..., Config] = Config.from_args,
    _HTTPServer: Callable[..., HTTPServer] = FidoHTTPServer,
    _make_registry: Callable[..., WorkerRegistry] = make_registry,
    _path_home: Callable[[], Path] = Path.home,
    _basic_config: Callable[..., None] = logging.basicConfig,
    _stderr: Any = sys.stderr,
    _populate_memberships: Callable[..., None] = populate_memberships,
    _signal: Callable[..., Any] = signal.signal,
    _kill_active_children: Callable[..., None] = kill_active_children,
    _Watchdog: type[Watchdog] = Watchdog,
    _ReconcileWatchdog: type[ReconcileWatchdog] = ReconcileWatchdog,
    _RateLimitMonitor: type[RateLimitMonitor] = RateLimitMonitor,
    _preflight_repo_identity: Callable[..., None] = preflight_repo_identity,
    _preflight_tools: Callable[..., None] = preflight_tools,
    _preflight_sub_dir: Callable[..., None] = preflight_sub_dir,
    _preflight_gh_auth: Callable[..., None] = preflight_gh_auth,
    _GitHub: type[GitHub] = GitHub,
    _bootstrap_issue_caches: Callable[..., None] = bootstrap_issue_caches,
) -> None:
    config = _from_args()

    repo_filter = RepoContextFilter()
    handlers: list[logging.Handler] = [logging.StreamHandler(_stderr)]
    for handler in handlers:
        handler.addFilter(repo_filter)

    _basic_config(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(repo_name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # Route uncaught exceptions through the logger so Docker/stdout captures
    # tracebacks through the same stream as normal runtime logs.
    def _log_uncaught(
        exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any
    ) -> None:
        log.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
        exc_info: Any = (args.exc_type, args.exc_value, args.exc_traceback)
        log.critical(
            "uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=exc_info,
        )

    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_exception

    infra = real_infra()
    WebhookHandler.infra = infra

    gh = _GitHub()
    try:
        _preflight_tools(infra.fs)
        _preflight_sub_dir(config, infra.fs)
        _preflight_gh_auth(gh)
        _preflight_repo_identity(config.repos, infra.proc)
    except PreflightError as e:
        raise SystemExit(str(e)) from e

    _populate_memberships(config, gh)

    WebhookHandler.config = config
    WebhookHandler.gh = gh
    WebhookHandler.static_files = StaticFiles(
        Path(__file__).resolve().parent / "static"
    )
    WebhookHandler.provider_factory = DefaultProviderFactory(
        session_system_file=config.sub_dir / "persona.md"
    )
    registry = _make_registry(config.repos, gh, config)
    WebhookHandler.registry = registry
    # Bootstrap issue caches eagerly so the picker has populated data immediately —
    # even for repos whose worker resumes on an existing issue and never calls
    # find_next_issue during this run (closes #837).
    _bootstrap_issue_caches(config.repos, gh, registry)
    # Route webhook-handler prompt calls through the per-repo persistent
    # ClaudeSession (closes #479 — "one claude per repo" invariant).
    provider.set_session_resolver(registry.get_session)
    _Watchdog(registry, config.repos).start_thread()
    _ReconcileWatchdog(registry, config.repos, gh).start_thread()
    rate_limit_monitor = _RateLimitMonitor(gh)
    rate_limit_monitor.start_thread()
    WebhookHandler.rate_limit_monitor = rate_limit_monitor
    WebhookHandler.fido_started_at = datetime.now(tz=timezone.utc)

    server = _HTTPServer(("", config.port), WebhookHandler)

    def _shutdown_handler(signum: int, _frame: object) -> None:
        log.info("fido received signal %d — terminating claude children", signum)
        _kill_active_children()
        server.server_close()
        sys.exit(0)

    _signal(signal.SIGTERM, _shutdown_handler)
    _signal(signal.SIGINT, _shutdown_handler)

    # Diagnostic hook: ``kill -USR1 <fido-pid>`` (or ``docker kill --signal=
    # SIGUSR1 fido``) dumps every thread's Python stack to stderr — captured
    # in fido.log via the launcher's redirect.  Lets us see exactly which
    # line a hung worker thread is parked on without needing pdb attached.
    # The :class:`faulthandler` module's signal handler is async-signal-safe
    # and compatible with the free-threaded (3.14t) runtime.
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    repos_str = ", ".join(f"{name}={rc.work_dir}" for name, rc in config.repos.items())
    log.info("fido listening on :%d — repos: %s", config.port, repos_str)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        _kill_active_children()
        server.server_close()
