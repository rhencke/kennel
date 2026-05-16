import dataclasses
import faulthandler
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
from types import TracebackType
from typing import IO, Any
from urllib.parse import urlparse

import requests
from frozendict import frozendict

from fido import provider
from fido.appstate import (
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
)
from fido.atomic import AtomicReader, create_atomic
from fido.claude import kill_active_children
from fido.config import Config, RepoConfig, RepoMembership
from fido.events import (
    FIDO_LOGINS,
    Action,
    Dispatcher,
    WebhookIngressOracle,
    create_task,
    launch_worker,
    queue_reply_tasks,
    reply_outcome_creates_tasks,
    reply_to_comment,
    reply_to_issue_comment,
    reply_to_review,
    thread_lineage_comment_ids,
)
from fido.github import GitHub, GraphQLError
from fido.infra import (
    Clock,
    Filesystem,
    Infra,
    ProcessRunner,
    real_infra,
)
from fido.provider import ThreadKind
from fido.provider_factory import DefaultProviderFactory
from fido.provider_pressure import ProviderPressureMonitor
from fido.rate_limit import RateLimitMonitor
from fido.registry import (
    WebhookActivityHandle,
    WorkerRegistry,
    make_registry,
)
from fido.rocq import self_restart as restart_fsm
from fido.session_lock_watchdog import SessionLockWatchdog
from fido.static_files import StaticFiles
from fido.store import FidoStore, ReplyPromiseRecord
from fido.synthesis_executor import CommentTarget, SynthesisExecutor
from fido.tasks import Tasks
from fido.watchdog import (
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

# Comment webhook event → the payload key carrying the item number that
# scopes the cache.  ``pull_request_review`` events nest their entry
# under ``payload["review"]`` (vs ``payload["comment"]`` for the other
# two), but the item-number lookup uses the parent regardless.
_COMMENT_EVENT_PARENT_KEYS: dict[str, str] = {
    "issue_comment": "issue",
    "pull_request_review_comment": "pull_request",
    "pull_request_review": "pull_request",
}


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


def _jsonable(value: object) -> object:
    """``json.dumps`` default for snapshot dataclasses.

    Walks frozen dataclasses, frozendict, and datetime — everything
    that lives in :class:`FidoState` — into JSON-friendly primitives
    so ``/status.json`` is a single, leaf-driven dump with zero
    flattening / compatibility code (#1696).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, frozendict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


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
    # Read face of the FidoState atomic cell — set by run() alongside
    # registry.  Status serialisation reads from here; registry only writes.
    state_reader: AtomicReader[FidoState]
    provider_factory: DefaultProviderFactory | None = None
    # Injectable collaborators — set as class attributes so HTTP-driven tests
    # can replace them without patching module-level names.  ``gh`` is
    # accessed through the property below so callers see ``GitHub`` rather
    # than ``GitHub | None``; ``serve()`` sets ``_gh`` before any handler
    # runs, so by the time the property fires it's guaranteed non-None.
    _gh: GitHub | None = None

    @property
    def gh(self) -> GitHub:
        gh = type(self)._gh
        if gh is None:
            raise RuntimeError(
                "FidoHTTPHandler.gh accessed before serve() initialised it"
            )
        return gh

    dispatchers: dict[str, Dispatcher] = {}
    # Infrastructure ports — set by server.run() composition root.
    infra: Infra = real_infra()
    static_files: StaticFiles | None = None
    _fn_reply_to_comment = staticmethod(reply_to_comment)
    _fn_reply_to_review = staticmethod(reply_to_review)
    _fn_reply_to_issue_comment = staticmethod(reply_to_issue_comment)
    _fn_create_task = staticmethod(create_task)
    _fn_launch_worker = staticmethod(launch_worker)
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
    # _restart_fsm_lock serialises all reads and writes of _restart_fsm_state
    # across ThreadingHTTPServer handler threads.  Python 3.14t has no GIL —
    # the read-modify-write in _restart_fsm_transition is not atomic without
    # an explicit lock.  Class-level so every handler instance shares it.
    _restart_fsm_lock: threading.Lock = threading.Lock()

    def _restart_fsm_transition(self, event: restart_fsm.Event) -> restart_fsm.State:
        """Fire *event* against the process-level self-restart FSM.

        Raises :exc:`AssertionError` if the transition is rejected — a
        restart-sequence violation crashes loudly rather than silently
        continuing in an undefined state.

        Uses ``type(self)`` so the class-level state is updated and visible
        to every subsequent handler instance in the same process.
        """
        with type(self)._restart_fsm_lock:
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

        # Route by repo — all keys below are schema-required on real GitHub
        # webhook payloads.  KeyError means malformed payload → 500 so GitHub
        # retries and the failure surfaces rather than silently routing to
        # "unregistered repo" or disabling self-restart.
        try:
            repo_name = payload["repository"]["full_name"]
            # Pre-compute self-restart triggers — needed for both registered and
            # unregistered repos (_self_restart verifies the runner's git remote).
            default_branch = payload["repository"]["default_branch"]
            is_pr_merged = (
                event == "pull_request"
                and payload["action"] == "closed"
                and bool(payload["pull_request"]["merged"])
            )
            is_default_push = (
                event == "push" and payload["ref"] == f"refs/heads/{default_branch}"
            )
        except KeyError as exc:
            log.exception(
                "webhook: malformed payload, missing key %s (event=%s delivery=%s)",
                exc,
                event,
                delivery,
            )
            self._respond(500, "malformed payload")
            return
        repo_cfg = self.config.repos.get(repo_name)

        log.info(
            "webhook: event=%s action=%s repo=%s delivery=%s",
            event,
            payload.get("action", "-"),
            repo_name,
            delivery,
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
        dispatcher = self.dispatchers[repo_name]
        try:
            action = dispatcher.dispatch(
                event,
                payload,
                delivery_id=delivery,
                oracle=type(self).ingress_oracle,
            )
        except Exception:
            log.exception("dispatch failed for %s", repo_name)
            self._respond(500, "dispatch error")
            return

        if action and self._action_records_durable_demand(action):
            self.registry.note_durable_demand(repo_cfg.name)
            self.registry.note_provider_interrupt_requested(repo_cfg.name)

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

        # Patch the per-(repo, item) comment cache for comment / review
        # events (#1754).  Same failure tolerance as the issue cache —
        # log, continue, the eventual reconcile (#1759) will heal.
        try:
            self._patch_comment_cache(event, payload, repo_cfg)
        except Exception:
            log.exception(
                "comment-cache patch failed for %s — next list fetch will heal",
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
            self._preempt_worker_best_effort(repo_cfg.name)
            self.registry.enter_untriaged(repo_cfg.name)

        # Process in background thread so we don't block the server.
        if action:
            type(self)._fn_spawn_bg(self._process_action, (action, repo_cfg))

    def _preempt_worker_best_effort(self, repo_name: str) -> None:
        """Try to interrupt the current worker after durable demand is recorded."""
        session = self.registry.get_session(repo_name)
        if session is None:
            return
        try:
            session.preempt_worker()
        except Exception as exc:
            if provider.is_recoverable_provider_wedge(exc):
                log.exception(
                    "provider preempt wedged for %s after durable webhook enqueue "
                    "— recovering provider",
                    repo_name,
                )
                recovered = self.registry.recover_provider(repo_name)
                if recovered:
                    log.warning(
                        "provider recovery requested for %s after preempt wedge",
                        repo_name,
                    )
                else:
                    log.error(
                        "provider recovery unavailable for %s after preempt wedge",
                        repo_name,
                    )
                return
            log.exception(
                "provider preempt failed for %s after durable webhook enqueue",
                repo_name,
            )

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

    def _patch_comment_cache(
        self, event: str, payload: dict[str, Any], repo_cfg: RepoConfig
    ) -> None:
        """Route a comment / review webhook to the per-(repo, item) cache.

        Extracts the item (issue or PR) number from the payload, looks
        up the right :class:`~fido.comment_cache.CommentCache` via the
        registry, and hands the event to ``apply_event``.  Unrelated
        event types are silently ignored (every webhook hits this
        method; only comment/review events have a cache target).

        ``issue_comment`` events fire for plain-issue comments AND PR
        top-level comments (PRs are issues for that endpoint).  Fido's
        dispatcher (``events.py``) ignores plain-issue comments, so
        we mirror that filter here — otherwise busy repos with lots of
        issue traffic accumulate caches we'll never use (codex P2 on
        #1751).  Proper lifecycle binding (cache created only when
        Fido has work on the item) lands in #1757.
        """
        parent_key = _COMMENT_EVENT_PARENT_KEYS.get(event)
        if parent_key is None:
            return
        parent = payload.get(parent_key)
        if not isinstance(parent, dict):
            return
        item_number = parent.get("number")
        if not isinstance(item_number, int):
            return
        if event == "issue_comment" and not parent.get("pull_request"):
            # Plain-issue comment — Dispatcher ignores these too.
            return
        cache = self.registry.get_comment_cache(repo_cfg.name, item_number, self.gh)
        cache.apply_event(event, payload)

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
        provider.set_thread_kind(ThreadKind.WEBHOOK)
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
                    with session.hold_for_handler():
                        # Publish immediately after acquiring the handler hold
                        # so repo_state.provider reflects the current ownership
                        # (clears any stale worker-owner from the last snapshot).
                        self.registry.publish_provider_snapshot(repo_cfg.name)
                        try:
                            self._process_action_inner(action, repo_cfg, activity)
                        finally:
                            # Publish again on the way out so the snapshot is
                            # fresh after the session returns to the worker.
                            self.registry.publish_provider_snapshot(repo_cfg.name)
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

    def _action_records_durable_demand(self, action: Action) -> bool:
        """True when dispatch already enqueued durable PR-comment demand."""
        return action.preempts_worker and action.thread is not None

    def _describe_action(self, action: Action) -> str:
        """Short label for status display — what this webhook handler is doing."""
        if action.reply_to:
            return "handling review comment"
        if action.review_comments:
            return "handling review thread"
        if action.comment_body:
            return "handling PR comment"
        if action.thread and action.preempts_worker:
            return "ingesting PR comment"
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
        thread = action.reply_to or action.thread
        comment_type, comment_id = promise_key
        promise = FidoStore(repo_cfg.work_dir).prepare_reply(
            owner="webhook",
            comment_type=comment_type,
            anchor_comment_id=comment_id,
            covered_comment_ids=thread_lineage_comment_ids(thread),
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
        gh = self.gh
        # Determine which comment (if any) will receive the eyes reaction.
        # Resolved before the try block so both the main path and the
        # recoverable-exception handler can reference _eyes_comment_info.
        _eyes_comment_info: dict[str, Any] | None = None
        if action.reply_to:
            _eyes_comment_info = action.reply_to
        elif action.thread:
            _eyes_comment_info = action.thread
        try:
            handled = False
            category: str | None = None
            titles: list[str] = []
            queued_tasks = 0

            # Unblock any BLOCKED tasks BEFORE the handler triggers a
            # background rescope (codex on #1738).  create_task() spawns
            # _reorder_tasks_background, which snapshots the task list
            # for validation; if the unblock ran AFTER create_task, a
            # fast rescope could validate against the stale blocked
            # state and reject a valid merge into what should now be a
            # pending target.  Doing the unblock here, before any
            # handler runs, makes the post-unblock state the snapshot
            # the rescope sees.
            if action.reply_to or action.comment_body or action.thread:
                Tasks(repo_cfg.work_dir).unblock_tasks()

            # Post eyes reaction immediately to signal work-in-progress.
            # Fires before any triage / rescope / reply work so the comment
            # author sees acknowledgement within ~1s of their comment (#1243).
            # Best-effort — a failed reaction must never abort the handler.
            # Skipped for bot-authored actions (don't react to other bots).
            #
            # Skipped for batch arrivals (#1662): if there's any other open
            # comment for this repo (pending or in-progress), this comment is
            # part of a batch — the worker posts eyes on claim instead, so at
            # most one comment per repo carries eyes at a time.  Solo comments
            # (no other open work) still get the sub-1s ack.
            _eyes_posted = False
            if not action.is_bot and _eyes_comment_info is not None:
                _eyes_repo = _eyes_comment_info.get("repo")
                _eyes_ctype = str(_eyes_comment_info.get("comment_type", "issues"))
                _eyes_cid = _eyes_comment_info.get("comment_id")
                if _eyes_repo and isinstance(_eyes_cid, int):
                    if FidoStore(repo_cfg.work_dir).has_other_open_pr_comments(
                        repo=_eyes_repo, exclude_comment_id=_eyes_cid
                    ):
                        log.debug(
                            "eager eyes skipped for comment %d — repo has "
                            "other open comments; worker posts eyes on claim",
                            _eyes_cid,
                        )
                    else:
                        try:
                            gh.add_reaction(_eyes_repo, _eyes_ctype, _eyes_cid, "eyes")
                            log.info("eyes reaction posted for comment %d", _eyes_cid)
                            _eyes_posted = True
                        except Exception:
                            log.exception(
                                "failed to post eyes reaction for comment %d "
                                "— continuing",
                                _eyes_cid,
                            )

            if action.reply_to:
                promise = self._prepare_reply(repo_cfg, action)
                if promise is None:
                    handled = True
                    category, titles = None, []
                    # Dedup skip — another handler owns this comment.  Clean up
                    # the eyes reaction posted above so it doesn't linger.
                    if _eyes_comment_info is not None:
                        self._remove_eyes_best_effort(gh, _eyes_comment_info)
                else:
                    activity.set_description("triaging review comment")
                    try:
                        category, titles = type(self)._fn_reply_to_comment(
                            action, self.config, repo_cfg, gh, self.registry
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
                        dispatcher=self.dispatchers[repo_cfg.name],
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
                    # Dedup skip — clean up the eyes reaction posted above.
                    if _eyes_comment_info is not None:
                        self._remove_eyes_best_effort(gh, _eyes_comment_info)
                else:
                    activity.set_description("triaging PR comment")
                    try:
                        category, titles = type(self)._fn_reply_to_issue_comment(
                            action, self.config, repo_cfg, gh, self.registry
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
                    dispatcher=self.dispatchers[repo_cfg.name],
                )

            log.info(
                "action outcome: handled=%s category=%s tasks=%d",
                handled,
                category,
                queued_tasks,
            )
            # Non-comment events just trigger fido worker — no task needed.
            # The unblock for comment events ran above, before handlers,
            # to close a race with the background rescope (codex on #1738).
            type(self)._fn_launch_worker(repo_cfg, self.registry)
        except provider.SessionLeakError:
            # Let the outer _process_action handler halt fido — we must not
            # swallow a leak into the generic "confused reaction" path below.
            raise
        except (
            requests.RequestException,
            GraphQLError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ):
            # Recoverable: transient GitHub/network failures and task-file I/O
            # contention.  Signal confused reaction so the author sees something
            # went wrong, then let this webhook thread exit cleanly.
            # Logic bugs (KeyError, TypeError, AttributeError, etc.) are NOT
            # caught here — they propagate and crash the thread loudly.
            log.exception("error processing action")
            if _eyes_comment_info is not None:
                self._remove_eyes_best_effort(gh, _eyes_comment_info)
            self._signal_action_error(action)

    def _remove_eyes_best_effort(
        self,
        gh: GitHub,
        eyes_comment_info: dict[str, Any],
    ) -> None:
        """Remove Fido's eyes reaction from a comment (best-effort).

        Builds a minimal :class:`~fido.synthesis_executor.CommentTarget` from
        *eyes_comment_info* and delegates to
        :meth:`~fido.synthesis_executor.SynthesisExecutor.remove_eyes_reaction`,
        which lists reactions, finds Fido's ``eyes`` entries, and deletes them.
        Called on all non-reply exit paths in :meth:`_process_action_inner` so
        the eyes reaction never lingers when no reply is posted.

        Silently no-ops if the required fields are absent.
        """
        repo = eyes_comment_info.get("repo")
        comment_id = eyes_comment_info.get("comment_id")
        comment_type = str(eyes_comment_info.get("comment_type", "issues"))
        if not repo or not isinstance(comment_id, int):
            return
        target = CommentTarget(
            repo=str(repo),
            pr=0,
            comment_id=comment_id,
            comment_type=comment_type,
        )
        SynthesisExecutor(gh, fido_logins=FIDO_LOGINS).remove_eyes_reaction(target)

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
            with type(self)._restart_fsm_lock:
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
        if self.path == "/status.json":
            body = json.dumps(self.state_reader.get(), default=_jsonable).encode()
            self._respond_body("application/json", body)
        elif self.path.startswith("/static/"):
            self._serve_static()
        else:
            self._respond(200, "fido is running")

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
        except (
            requests.RequestException,
            GraphQLError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            # Transient GitHub/network failure — ReconcileWatchdog heals within
            # the hour.  Auth errors (RuntimeError) and logic bugs are NOT
            # caught; they crash startup loud so misconfiguration is visible.
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
    _stderr: IO[str] = sys.stderr,
    _populate_memberships: Callable[..., None] = populate_memberships,
    _signal: Callable[..., Any] = signal.signal,
    _kill_active_children: Callable[..., None] = kill_active_children,
    _Watchdog: type[Watchdog] = Watchdog,
    _ReconcileWatchdog: type[ReconcileWatchdog] = ReconcileWatchdog,
    _SessionLockWatchdog: type[SessionLockWatchdog] = SessionLockWatchdog,
    _RateLimitMonitor: type[RateLimitMonitor] = RateLimitMonitor,
    _ProviderPressureMonitor: type[ProviderPressureMonitor] = ProviderPressureMonitor,
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
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
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
    # Composition root sets the class-level lazy collaborator; the proper
    # constructor-DI redo is tracked in #1762.
    WebhookHandler._gh = gh  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    dispatchers = {
        name: Dispatcher(config, repo_cfg, gh)
        for name, repo_cfg in config.repos.items()
    }
    WebhookHandler.dispatchers = dispatchers
    WebhookHandler.static_files = StaticFiles(
        Path(__file__).resolve().parent / "static"
    )
    WebhookHandler.provider_factory = DefaultProviderFactory(
        session_system_file=config.sub_dir / "persona.md"
    )
    # Create the atomic FidoState cell here (composition root) and hand the
    # two faces to their respective owners: the updater goes to the registry
    # (write-only) and to the rate-limit monitor; the reader stays here and
    # is placed on WebhookHandler so the status serialisation path can read
    # without going through the registry at all.
    process_started_at = datetime.now(tz=timezone.utc)
    state_reader, state_updater = create_atomic(
        FidoState(
            repos=frozendict(),
            github_limits=_ZERO_GITHUB_LIMITS,
            process_started_at=process_started_at,
        )
    )
    registry = _make_registry(
        config.repos, gh, config, dispatchers=dispatchers, state_updater=state_updater
    )
    WebhookHandler.registry = registry
    WebhookHandler.state_reader = state_reader
    # Bootstrap issue caches eagerly so the picker has populated data immediately —
    # even for repos whose worker resumes on an existing issue and never calls
    # find_next_issue during this run (closes #837).
    _bootstrap_issue_caches(config.repos, gh, registry)
    # Route webhook-handler prompt calls through the per-repo persistent
    # ClaudeSession (closes #479 — "one claude per repo" invariant).
    provider.set_session_resolver(registry.get_session)
    _Watchdog(registry, config.repos).start_thread()
    _ReconcileWatchdog(registry, config.repos, gh).start_thread()
    # Session-lock watchdog evicts FSM lock holders that have parked past
    # the deadline — without it, a holder wedged inside
    # ``consume_until_result`` on a streaming-forever subprocess holds
    # the lock indefinitely (closes #1377).
    _SessionLockWatchdog(registry, config.repos).start_thread()
    _RateLimitMonitor(gh, state_updater).start_thread()
    _ProviderPressureMonitor(
        config.repos, state_updater, WebhookHandler.provider_factory
    ).start_thread()

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
