"""kennel status — reads per-repo fido state and formats a one-shot summary."""

from __future__ import annotations

import fcntl
import json
import logging
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kennel.color import (
    BOLD,
    CYAN,
    DARK_GRAY,
    DIM,
    GREEN,
    GREEN_BG,
    MAGENTA,
    RED,
    RED_BOLD,
    YELLOW,
    YELLOW_BG,
    color,
    color_enabled,
    rgb_bg,
    rgb_fg,
    wrap_bg_line,
    wrap_raw,
)
from kennel.config import RepoConfig
from kennel.provider import (
    ProviderID,
    ProviderPressureStatus,
    palette_for,
)
from kennel.provider_factory import DefaultProviderFactory
from kennel.state import State
from kennel.tasks import Tasks

log = logging.getLogger(__name__)


@dataclass
class WebhookActivityInfo:
    """Lightweight in-flight webhook handler, used purely for display.

    *thread_id* matches the ``thread_id`` stored on
    :class:`~kennel.provider.SessionTalker` so display can identify which
    webhook (if any) is currently driving claude.
    """

    description: str
    elapsed_seconds: int
    thread_id: int = 0


@dataclass
class ClaudeTalkerInfo:
    """Who is currently driving this repo's claude subprocess."""

    thread_id: int
    kind: str  # "worker" | "webhook"
    description: str
    claude_pid: int


@dataclass
class RateLimitWindowInfo:
    """One ``/rate_limit`` resource window for ``kennel status`` display."""

    name: str
    used: int
    limit: int
    resets_at: datetime


@dataclass
class RateLimitInfo:
    """Latest GitHub rate-limit snapshot for ``kennel status`` display
    (closes #812 follow-up)."""

    rest: RateLimitWindowInfo
    graphql: RateLimitWindowInfo
    fetched_at: datetime


@dataclass
class IssueCacheInfo:
    """Snapshot of the per-repo issue tree cache for ``kennel status`` display
    (closes #812).

    Mirrors the fields on :class:`~kennel.issue_cache.CacheMetrics` but
    keeps display-only here so the status module does not depend on the
    cache module's class identity.
    """

    loaded: bool
    open_issues: int
    events_applied: int
    events_dropped_stale: int
    last_event_at: datetime | None
    last_reconcile_at: datetime | None
    last_reconcile_drift: int


@dataclass
class RepoStatus:
    name: str
    fido_running: bool
    issue: int | None
    pending: int
    completed: int
    current_task: str | None  # title of first in_progress or pending task
    claude_pid: int | None
    claude_uptime: int | None  # seconds
    worker_what: str | None  # activity text from the live registry
    crash_count: int  # number of unexpected worker deaths since kennel started
    last_crash_error: str | None  # error from the most recent crash, if any
    worker_stuck: bool  # True if the worker is alive but making no progress
    provider: ProviderID = ProviderID.CLAUDE_CODE
    provider_status: ProviderPressureStatus | None = None
    # Newer fields default so callers (tests) can omit them safely.
    issue_title: str | None = None
    issue_elapsed_seconds: int | None = None
    pr_number: int | None = None
    pr_title: str | None = None
    task_number: int | None = None  # active task position within the current queue
    task_total: int | None = None  # current queue length (pending + in_progress)
    worker_uptime: int | None = None
    webhook_activities: list[WebhookActivityInfo] = field(default_factory=list)
    session_owner: str | None = None
    session_alive: bool = False
    session_dropped_count: int = 0
    claude_talker: ClaudeTalkerInfo | None = None
    rescoping: bool = False  # True while a background Opus reorder is in flight
    issue_cache: IssueCacheInfo | None = None


@dataclass
class KennelStatus:
    kennel_pid: int | None
    kennel_uptime: int | None  # seconds
    repos: list[RepoStatus]
    provider_statuses: list[ProviderPressureStatus] = field(default_factory=list)
    rate_limit: RateLimitInfo | None = None


def _format_uptime(seconds: int) -> str:
    """Format seconds as a compact human-readable string (e.g. '2h13m', '45s')."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if mins:
        return f"{hours}h{mins}m"
    return f"{hours}h"


def _fido_running(lock_path: Path) -> bool:
    """Return True if the fido lock file is held by another process."""
    if not lock_path.exists():
        return False
    try:
        fd = open(lock_path)  # noqa: SIM115
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
        finally:
            fd.close()
    except OSError:
        return False


def _pgrep(pattern: str, *, _run: Callable[..., Any] = subprocess.run) -> list[int]:
    """Return PIDs whose command line matches pattern via pgrep -f.

    Best-effort enrichment for status display.  A nonzero exit code (e.g.
    pgrep exits 1 when no processes match) is not treated as an error —
    the PID list is built from stdout alone.  Returns [] on OSError
    (e.g. pgrep not installed).
    """
    try:
        result = _run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
        )
        pids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    pids.append(int(line))
                except ValueError:
                    pass
        return pids
    except OSError:
        return []


def _process_uptime_seconds(
    pid: int, *, _run: Callable[..., Any] = subprocess.run
) -> int | None:
    """Return elapsed seconds since the process started, or None if unavailable.

    Best-effort enrichment for status display.  A nonzero exit code (e.g.
    ps exits non-zero for an unknown PID) is not treated as an error —
    callers receive None whenever the value cannot be determined.
    """
    try:
        result = _run(
            ["ps", "-p", str(pid), "-o", "etimes="],
            capture_output=True,
            text=True,
        )
        text = result.stdout.strip()
        if text:
            return int(text)
    except (OSError, ValueError):  # fmt: skip
        pass
    return None


def _repos_from_pid(pid: int) -> list[RepoConfig]:
    """Read repo specs from /proc/<pid>/cmdline, returning [] if unavailable."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    repos = []
    for arg in cmdline.rstrip(b"\x00").split(b"\x00"):
        try:
            spec = arg.decode()
        except UnicodeDecodeError:
            continue
        if ":" not in spec:
            continue
        name, remainder = spec.split(":", 1)
        if ":" not in remainder:
            continue
        path_str, provider_str = remainder.rsplit(":", 1)
        try:
            provider = ProviderID(provider_str)
        except ValueError:
            continue
        if "/" not in name:
            continue
        repos.append(
            RepoConfig(
                name=name,
                work_dir=Path(path_str).expanduser(),
                provider=provider,
            )
        )
    return repos


def _kennel_pid() -> int | None:
    """Return the PID of the running kennel server, or None."""
    pids = _pgrep("kennel --port")
    return pids[0] if pids else None


def running_repo_configs(
    *,
    _kennel_pid_fn: Callable[[], int | None] = _kennel_pid,
    _repos_from_pid_fn: Callable[[int], list[RepoConfig]] = _repos_from_pid,
) -> list[RepoConfig]:
    """Return the repo configs for the currently running kennel, or []."""
    pid = _kennel_pid_fn()
    if pid is None:
        return []
    return _repos_from_pid_fn(pid)


def _status_persona_path() -> Path:
    return Path(__file__).resolve().parents[1] / "sub" / "persona.md"


def provider_statuses_for_repo_configs(
    repo_configs: list[RepoConfig],
    *,
    _provider_factory: DefaultProviderFactory | None = None,
) -> dict[ProviderID, ProviderPressureStatus]:
    """Return one normalized provider-pressure summary per configured provider."""
    provider_factory = _provider_factory or DefaultProviderFactory(
        session_system_file=_status_persona_path()
    )
    statuses: dict[ProviderID, ProviderPressureStatus] = {}
    for repo_cfg in repo_configs:
        if repo_cfg.provider in statuses:
            continue
        statuses[repo_cfg.provider] = ProviderPressureStatus.from_snapshot(
            provider_factory.create_api(repo_cfg).get_limit_snapshot()
        )
    return statuses


def _parse_provider_status(raw: object) -> ProviderPressureStatus | None:
    """Parse one serialized provider-pressure record from /status.json."""
    if not isinstance(raw, dict):
        return None
    try:
        provider = ProviderID(raw["provider"])
    except KeyError, TypeError, ValueError:
        return None
    resets_at = raw.get("resets_at")
    try:
        parsed_reset = (
            datetime.fromisoformat(resets_at) if isinstance(resets_at, str) else None
        )
    except ValueError:
        parsed_reset = None
    pressure = raw.get("pressure")
    return ProviderPressureStatus(
        provider=provider,
        window_name=raw.get("window_name")
        if isinstance(raw.get("window_name"), str)
        else None,
        pressure=float(pressure) if isinstance(pressure, int | float) else None,
        resets_at=parsed_reset,
        unavailable_reason=raw.get("unavailable_reason")
        if isinstance(raw.get("unavailable_reason"), str)
        else None,
    )


def _parse_iso_datetime(raw: object) -> datetime | None:
    """Parse an ISO-8601 string into a tz-aware datetime, or ``None``."""
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _parse_issue_cache(raw: object) -> IssueCacheInfo | None:
    """Parse one serialized issue-cache record from /status.json (#812)."""
    if not isinstance(raw, dict):
        return None
    return IssueCacheInfo(
        loaded=bool(raw.get("loaded", False)),
        open_issues=int(raw.get("open_issues", 0)),
        events_applied=int(raw.get("events_applied", 0)),
        events_dropped_stale=int(raw.get("events_dropped_stale", 0)),
        last_event_at=_parse_iso_datetime(raw.get("last_event_at")),
        last_reconcile_at=_parse_iso_datetime(raw.get("last_reconcile_at")),
        last_reconcile_drift=int(raw.get("last_reconcile_drift", 0)),
    )


def _port_from_pid(pid: int) -> int | None:
    """Extract the --port value from kennel's /proc/<pid>/cmdline, or None."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    args = cmdline.rstrip(b"\x00").split(b"\x00")
    for i, arg in enumerate(args):
        if arg == b"--port" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


def _fetch_activities(
    port: int, *, _urlopen: Callable[..., Any] = urllib.request.urlopen
) -> tuple[dict[str, dict[str, Any]], RateLimitInfo | None]:
    """Query GET /status.json for live worker activity, provider pressure,
    and the global GitHub rate-limit snapshot (closes #812 follow-up).

    Returns ``({}, None)`` on any fetch/parse failure — ``kennel status``
    treats empty as "kennel down" and renders accordingly.
    """
    try:
        with _urlopen(f"http://localhost:{port}/status.json", timeout=2) as resp:
            data = json.loads(resp.read())
        activities_raw = data.get("activities", [])
        rate_limit_raw = data.get("rate_limit")
        activities = {
            item["repo_name"]: {
                "what": item["what"],
                "crash_count": item["crash_count"],
                "last_crash_error": item["last_crash_error"],
                "is_stuck": item.get("is_stuck", False),
                "worker_uptime_seconds": item.get("worker_uptime_seconds"),
                "webhook_activities": item.get("webhook_activities", []),
                "session_owner": item.get("session_owner"),
                "session_alive": item.get("session_alive", False),
                "session_pid": item.get("session_pid"),
                "session_dropped_count": item.get("session_dropped_count", 0),
                "claude_talker": item.get("claude_talker"),
                "provider_status": _parse_provider_status(item.get("provider_status")),
                "rescoping": item.get("rescoping", False),
                "issue_cache": item.get("issue_cache"),
            }
            for item in activities_raw
            if "repo_name" in item and "what" in item
        }
        return activities, _parse_rate_limit(rate_limit_raw)
    except Exception:  # noqa: BLE001
        return {}, None


def _parse_rate_limit(raw: object) -> RateLimitInfo | None:
    """Parse the ``rate_limit`` field from /status.json into a
    :class:`RateLimitInfo`, or ``None`` when missing/malformed."""
    if not isinstance(raw, dict):
        return None
    rest_raw = raw.get("rest")
    graphql_raw = raw.get("graphql")
    fetched_at = _parse_iso_datetime(raw.get("fetched_at"))
    if not isinstance(rest_raw, dict) or not isinstance(graphql_raw, dict):
        return None
    if fetched_at is None:
        return None
    return RateLimitInfo(
        rest=_parse_rate_window(rest_raw),
        graphql=_parse_rate_window(graphql_raw),
        fetched_at=fetched_at,
    )


def _parse_rate_window(raw: dict[str, Any]) -> RateLimitWindowInfo:
    resets_at = _parse_iso_datetime(raw.get("resets_at"))
    if resets_at is None:
        resets_at = datetime.fromtimestamp(0, tz=timezone.utc)
    return RateLimitWindowInfo(
        name=str(raw.get("name", "")),
        used=int(raw.get("used", 0)),
        limit=int(raw.get("limit", 0)),
        resets_at=resets_at,
    )


def _claude_pid(fido_dir: Path) -> int | None:
    """Return the PID of the claude process for this fido_dir, or None.

    Matches both initial sessions (command line includes the system-prompt
    file under fido_dir/system) and resumed sessions (command line includes
    ``--resume <session_id>`` from state.json, which doesn't mention the
    system file).
    """
    pids = _pgrep(str(fido_dir / "system"))
    if pids:
        return pids[0]
    session_id = State(fido_dir).load().get("setup_session_id")
    if not session_id:
        return None
    pids = _pgrep(session_id)
    return pids[0] if pids else None


def _git_dir(
    work_dir: Path, *, _run: Callable[..., Any] = subprocess.run
) -> Path | None:
    """Return the absolute git directory for work_dir, or None if unavailable.

    Best-effort enrichment for status display.  Returns None on nonzero
    exit (not a git repo) or if git is not installed.
    """
    try:
        result = _run(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError, FileNotFoundError:
        return None


def _read_state(fido_dir: Path) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]  # used by tests
    """Read state.json from fido_dir, returning {} if absent or unreadable."""
    path = fido_dir / "state.json"
    lock_path = fido_dir / "state.lock"
    try:
        lock_path.touch(exist_ok=True)
        with open(lock_path) as lock_fd:  # noqa: SIM115
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            if not path.exists():
                return {}
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return {}


def _current_task(task_list: list[dict[str, Any]]) -> str | None:
    """Return the title of the first in_progress task, then the first pending task."""
    for t in task_list:
        if t["status"] == "in_progress":
            return t["title"]
    for t in task_list:
        if t["status"] == "pending":
            return t["title"]
    return None


def _task_position(task_list: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Return (task_number, total) for display.

    Reports the active task's position within the current queue (pending +
    in-progress tasks only). Completed historical tasks do not offset the
    position because status should describe what remains to do right now.
    Returns (None, None) when there are no non-completed tasks.
    """
    non_completed = [t for t in task_list if t["status"] != "completed"]
    if not non_completed:
        return (None, None)
    total = len(non_completed)
    for idx, t in enumerate(non_completed, start=1):
        if t["status"] == "in_progress":
            return (idx, total)
    return (1, total)


def _elapsed_since_iso(
    iso: str | None, *, _now: Callable[[], datetime] | None = None
) -> int | None:
    """Return integer seconds since *iso* (an ISO-8601 timestamp), or None."""
    if not iso:
        return None
    try:
        started = datetime.fromisoformat(iso)
    except TypeError, ValueError:
        return None
    now = _now() if _now is not None else datetime.now(tz=timezone.utc)
    return max(0, int((now - started).total_seconds()))


def repo_status(
    repo_config: RepoConfig,
    worker_what: str | None = None,
    crash_count: int = 0,
    last_crash_error: str | None = None,
    worker_stuck: bool = False,
    worker_uptime: int | None = None,
    webhook_activities: list[WebhookActivityInfo] | None = None,
    session_owner: str | None = None,
    session_alive: bool = False,
    session_pid: int | None = None,
    session_dropped_count: int = 0,
    claude_talker: ClaudeTalkerInfo | None = None,
    rescoping: bool = False,
    provider_status: ProviderPressureStatus | None = None,
    issue_cache: IssueCacheInfo | None = None,
) -> RepoStatus:
    """Collect status for a single repo."""
    webhook_activities = list(webhook_activities or [])
    git_dir = _git_dir(repo_config.work_dir)
    if git_dir is None:
        return RepoStatus(
            name=repo_config.name,
            fido_running=False,
            issue=None,
            issue_title=None,
            issue_elapsed_seconds=None,
            pr_number=None,
            pr_title=None,
            pending=0,
            completed=0,
            current_task=None,
            task_number=None,
            task_total=None,
            claude_pid=None,
            claude_uptime=None,
            worker_uptime=worker_uptime,
            worker_what=worker_what,
            crash_count=crash_count,
            last_crash_error=last_crash_error,
            worker_stuck=worker_stuck,
            provider=repo_config.provider,
            provider_status=provider_status,
            webhook_activities=webhook_activities,
            session_owner=session_owner,
            session_alive=session_alive,
            session_dropped_count=session_dropped_count,
            claude_talker=claude_talker,
            rescoping=rescoping,
            issue_cache=issue_cache,
        )

    fido_dir = git_dir / "fido"
    running = _fido_running(fido_dir / "lock")

    state = State(fido_dir).load()
    issue = state.get("issue")
    issue_title = state.get("issue_title")
    issue_elapsed = _elapsed_since_iso(state.get("issue_started_at"))
    pr_number = state.get("pr_number")
    pr_title = state.get("pr_title")

    task_list = Tasks(repo_config.work_dir).list()
    pending = sum(1 for t in task_list if t["status"] == "pending")
    completed = sum(1 for t in task_list if t["status"] == "completed")

    current = _current_task(task_list)
    task_number, task_total = _task_position(task_list)

    # Prefer the kennel-tracked session pid (authoritative, known from the
    # ClaudeSession subprocess) over a pgrep heuristic that can't find the
    # persistent session (its system file is outside fido_dir).
    claude_pid = session_pid if session_pid is not None else _claude_pid(fido_dir)
    claude_uptime = (
        _process_uptime_seconds(claude_pid) if claude_pid is not None else None
    )

    return RepoStatus(
        name=repo_config.name,
        fido_running=running,
        issue=issue,
        issue_title=issue_title,
        issue_elapsed_seconds=issue_elapsed,
        pr_number=pr_number,
        pr_title=pr_title,
        pending=pending,
        completed=completed,
        current_task=current,
        task_number=task_number,
        task_total=task_total,
        claude_pid=claude_pid,
        claude_uptime=claude_uptime,
        worker_uptime=worker_uptime,
        worker_what=worker_what,
        crash_count=crash_count,
        last_crash_error=last_crash_error,
        worker_stuck=worker_stuck,
        provider=repo_config.provider,
        provider_status=provider_status,
        webhook_activities=webhook_activities,
        session_owner=session_owner,
        session_alive=session_alive,
        session_dropped_count=session_dropped_count,
        claude_talker=claude_talker,
        rescoping=rescoping,
        issue_cache=issue_cache,
    )


def collect() -> KennelStatus:
    """Collect the full kennel + per-repo status."""
    pid = _kennel_pid()
    uptime = _process_uptime_seconds(pid) if pid is not None else None
    repo_configs = _repos_from_pid(pid) if pid is not None else []

    activities: dict[str, Any] = {}
    rate_limit_info: RateLimitInfo | None = None
    if pid is not None:
        port = _port_from_pid(pid)
        if port is not None:
            activities, rate_limit_info = _fetch_activities(port)

    provider_statuses: dict[ProviderID, ProviderPressureStatus] = {}
    repos = []
    for rc in repo_configs:
        info = activities.get(rc.name)
        webhook_list = []
        worker_uptime_val: int | None = None
        talker_info: ClaudeTalkerInfo | None = None
        session_pid_val: int | None = None
        if info:
            for w in info.get("webhook_activities") or []:
                webhook_list.append(
                    WebhookActivityInfo(
                        description=w["description"],
                        elapsed_seconds=int(w["elapsed_seconds"]),
                        thread_id=int(w.get("thread_id", 0)),
                    )
                )
            wu = info.get("worker_uptime_seconds")
            worker_uptime_val = int(wu) if wu is not None else None
            talker_raw = info.get("claude_talker")
            if talker_raw is not None:
                talker_info = ClaudeTalkerInfo(
                    thread_id=int(talker_raw["thread_id"]),
                    kind=talker_raw["kind"],
                    description=talker_raw["description"],
                    claude_pid=int(talker_raw["claude_pid"]),
                )
            sp = info.get("session_pid")
            session_pid_val = int(sp) if sp is not None else None
        cache_info = _parse_issue_cache(info.get("issue_cache")) if info else None
        provider_status = info.get("provider_status") if info else None
        if (
            isinstance(provider_status, ProviderPressureStatus)
            and provider_status.provider not in provider_statuses
        ):
            provider_statuses[provider_status.provider] = provider_status
        repos.append(
            repo_status(
                rc,
                worker_what=info["what"] if info else None,
                crash_count=info["crash_count"] if info else 0,
                last_crash_error=info["last_crash_error"] if info else None,
                worker_stuck=info["is_stuck"] if info else False,
                worker_uptime=worker_uptime_val,
                webhook_activities=webhook_list,
                session_owner=info.get("session_owner") if info else None,
                session_alive=bool(info.get("session_alive")) if info else False,
                session_pid=session_pid_val,
                session_dropped_count=int(info.get("session_dropped_count", 0))
                if info
                else 0,
                claude_talker=talker_info,
                rescoping=bool(info.get("rescoping")) if info else False,
                provider_status=provider_status,
                issue_cache=cache_info,
            )
        )
    return KennelStatus(
        kennel_pid=pid,
        kennel_uptime=uptime,
        repos=repos,
        provider_statuses=list(provider_statuses.values()),
        rate_limit=rate_limit_info,
    )


_SILENT_DISPLAY_THRESHOLD: int = 300  # seconds of claude silence before we note it


_WEBHOOK_DISPLAY_CAP: int = 5
"""Max webhook lines to print per repo; overflow rolled into a +N-more line."""


def _format_agent_line(repo: RepoStatus) -> str | None:
    """Dedicated first body line showing agent session state.

    Returns None when there is nothing to display (no pid, session not alive).
    Always shown when the agent exists — visible regardless of who is currently
    talking so the info is never buried.  Uses the repo's provider as the label
    so this works for Claude, Copilot, or any future provider.
    """
    if repo.claude_pid is None and not repo.session_alive:
        return None

    parts: list[str] = []
    if repo.claude_uptime is not None:
        parts.append(color(DIM, f"running {_format_uptime(repo.claude_uptime)}"))
    if (
        repo.session_alive
        and not _worker_is_agent_talker(repo)
        and repo.claude_talker is None
    ):
        parts.append(color(DIM, "session idle"))
    if repo.session_dropped_count > 0:
        noun = "session" if repo.session_dropped_count == 1 else "sessions"
        parts.append(color(RED_BOLD, f"dropped {noun} {repo.session_dropped_count}"))

    pid_str = (
        color(DIM, f"pid {repo.claude_pid}")
        if repo.claude_pid is not None
        else color(DIM, "agent")
    )
    label = color(BOLD, f"{repo.provider}:")
    if parts:
        joined = ", ".join(parts)
        return f"  {label} {pid_str} {color(DIM, '(')}{joined}{color(DIM, ')')}"
    return f"  {label} {pid_str}"


def _format_reset_at(resets_at: datetime) -> str:
    """Format provider reset times in a compact UTC form."""
    return resets_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _format_duration_until(target: datetime, *, now: datetime | None = None) -> str:
    """Render the gap between *now* and *target* as a compact ``Xh Ym`` /
    ``Ym`` / ``Ns`` string (closes #812 follow-up).

    Past times render as ``now`` — rate-limit windows that already reset
    aren't worth a negative-duration display.
    """
    current = now if now is not None else datetime.now(tz=timezone.utc)
    delta = (target - current).total_seconds()
    if delta <= 0:
        return "now"
    return _format_uptime(int(delta))


def _rate_limit_color(window: RateLimitWindowInfo) -> str:
    """Pick a color style for a rate-limit window (closes #812 follow-up).

    - Red when ≤ 5% remaining (danger — about to be throttled).
    - Yellow when ≤ 25% remaining (orange-ish in most terminals).
    - DIM otherwise (steady state, no warning needed).
    """
    pct_remaining = (
        100.0 * max(window.limit - window.used, 0) / window.limit
        if window.limit > 0
        else 0.0
    )
    if pct_remaining <= 5.0:
        return RED_BOLD
    if pct_remaining <= 25.0:
        return YELLOW
    return DIM


def _format_rate_limit_window(window: RateLimitWindowInfo, label: str) -> str:
    """Format one rate-limit window as ``LABEL used/limit (Xh Ym to reset)``
    with color coding by remaining percentage."""
    style = _rate_limit_color(window)
    rest_str = f"{window.used}/{window.limit}"
    reset_str = _format_duration_until(window.resets_at)
    body = f"{label} {rest_str} ({reset_str} to reset)"
    return color(style, body)


def _format_rate_limit_line(rate_limit: RateLimitInfo | None) -> str | None:
    """Top-of-status one-liner for the GitHub rate-limit snapshot.

    Hidden when the monitor hasn't yet completed its first refresh —
    avoids showing an empty placeholder during the few seconds between
    kennel boot and the first poll.
    """
    if rate_limit is None:
        return None
    rest_part = _format_rate_limit_window(rate_limit.rest, "REST")
    gql_part = _format_rate_limit_window(rate_limit.graphql, "GraphQL")
    label = color(BOLD, "GitHub:")
    return f"{label} {rest_part}, {gql_part}"


def _format_cache_line(repo: RepoStatus) -> str | None:
    """One-line picker-cache summary (#812).

    Hidden when the cache hasn't been bootstrapped — the line would just
    be noise for a repo that hasn't run a worker iteration yet.
    """
    cache = repo.issue_cache
    if cache is None or not cache.loaded:
        return None
    parts: list[str] = [color(DIM, f"{cache.open_issues} open")]
    parts.append(color(DIM, f"applied {cache.events_applied}"))
    if cache.events_dropped_stale:
        parts.append(color(DIM, f"stale-dropped {cache.events_dropped_stale}"))
    if cache.last_reconcile_at is not None:
        parts.append(color(DIM, f"reconciled drift {cache.last_reconcile_drift}"))
    label = color(BOLD, "Cache:")
    return f"  {label}  {', '.join(parts)}"


def _worker_is_agent_talker(repo: RepoStatus) -> bool:
    """True when the worker thread currently owns the provider session."""
    if repo.claude_talker is not None:
        return repo.claude_talker.kind == "worker"
    owner = repo.session_owner or ""
    return owner.startswith("worker-")


def _provider_status_style(status: ProviderPressureStatus) -> str:
    if status.paused:
        return DARK_GRAY
    if status.warning or status.level == "unavailable":
        return DIM
    return ""


def _provider_status_summary(status: ProviderPressureStatus) -> str:
    provider = str(status.provider)
    if status.unavailable_reason is not None:
        return f"{provider} unavailable"
    if status.percent_used is None:
        return f"{provider} limits unknown"
    details: list[str] = []
    if status.window_name:
        details.append(status.window_name.replace("_", " "))
    if status.resets_at is not None:
        details.append(f"resets {_format_reset_at(status.resets_at)}")
    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{provider} {status.percent_used}%{detail_text}"


def _styled_provider_status(status: ProviderPressureStatus) -> str:
    base_style = _provider_status_style(status)
    summary = _provider_status_summary(status)
    if base_style:
        # Preserve the existing dim/dark-gray rules (used when the provider
        # is paused or warning) — those signal state more than identity, so
        # they win over the provider-color highlight.
        return color(base_style, summary)
    # Highlight the provider name (prefix of the summary) with the provider's
    # bright fg so each provider is visually identifiable in the limits line.
    palette = palette_for(status.provider)
    provider_token = str(status.provider)
    if palette is not None and summary.startswith(provider_token):
        tail = summary[len(provider_token) :]
        return wrap_raw(rgb_fg(*palette.bright_fg), provider_token) + tail
    return summary


def _styled_repo_provider(repo: RepoStatus) -> str:
    """Render the repo's provider label without repeating global limits details."""
    provider_str = str(repo.provider)
    provider_status = repo.provider_status
    if provider_status is not None:
        base_style = _provider_status_style(provider_status)
        if base_style:
            # Warning/paused state wins over identity color — same precedence
            # as the global limits line in _styled_provider_status.
            return color(base_style, provider_str)
    palette = palette_for(repo.provider)
    if palette is not None:
        return wrap_raw(rgb_fg(*palette.bright_fg), provider_str)
    return provider_str


def _format_provider_summary_line(statuses: list[ProviderPressureStatus]) -> str | None:
    if not statuses:
        return None
    ordered = sorted(statuses, key=lambda status: status.provider.value)
    rendered = " | ".join(_styled_provider_status(status) for status in ordered)
    return f"{color(BOLD, 'limits:')} {rendered}"


def _format_repo_header(repo: RepoStatus) -> str:
    """Top line per repo: ``<name>: fido <state> — <stats>[ → <claude>]``.

    Stats list is comma-separated and only shows what matters right now:
    ``crashes N`` (skipped when 0), ``up X`` (worker thread uptime), ``BUSY``
    (no activity for ≥5m, informational since #444), optional ``last crash``
    line when crash_count > 0.  If nobody is currently talking to the agent,
    the generic pid/uptime suffix appears on this line.
    """
    state_word = "running" if repo.fido_running else "waiting"
    state_style = GREEN if repo.fido_running else DIM
    stats: list[str] = [_styled_repo_provider(repo)]
    if repo.crash_count > 0:
        stats.append(color(RED_BOLD, f"crashes {repo.crash_count}"))
    if repo.worker_uptime is not None:
        stats.append(color(DIM, f"up {_format_uptime(repo.worker_uptime)}"))
    if repo.worker_stuck:
        stats.append(color(RED, "BUSY"))
    if repo.crash_count > 0 and repo.last_crash_error:
        stats.append(color(RED_BOLD, f"last crash: {repo.last_crash_error}"))

    name_styled = color(BOLD, f"{repo.name}:")
    state_styled = color(state_style, f"fido {state_word}")
    header = f"{name_styled} {state_styled}"
    if stats:
        header += " — " + ", ".join(stats)
    return header


def _format_repo_body(repo: RepoStatus) -> list[str]:
    """Per-repo body lines in fixed order:

    1. ``{provider}: pid N (running Xm, session idle)`` — agent session state,
       only when a pid or live session is present; uses the configured provider
       name so it works for Claude, Copilot, or any future provider.
    2. ``Issue:  #N — title  (elapsed Xm)``
    3. ``PR:     #N — title``
    4. ``Worker: <state>`` (idle / task N/M — title / waiting on …)
    5. Webhook threads (indented ``├─`` / ``└─``), up to
       :data:`_WEBHOOK_DISPLAY_CAP`; a webhook currently talking to the agent
       sorts to the top and gets an ANSI background-highlighted label; overflow
       rolled into ``+N more webhook(s)``.
    """
    body: list[str] = []

    agent_line = _format_agent_line(repo)
    if agent_line is not None:
        body.append(agent_line)

    cache_line = _format_cache_line(repo)
    if cache_line is not None:
        body.append(cache_line)

    if repo.issue is None:
        body.append("  no assigned issues")
        return body

    issue_line = f"  {color(BOLD, 'Issue:')} {color(CYAN, f'#{repo.issue}')}"
    if repo.issue_title:
        issue_line += f" {color(DIM, '—')} {repo.issue_title}"
    if repo.issue_elapsed_seconds is not None:
        issue_line += (
            f" {color(DIM, f'(elapsed {_format_uptime(repo.issue_elapsed_seconds)})')}"
        )
    body.append(issue_line)

    if repo.pr_number is not None:
        pr_line = f"  {color(BOLD, 'PR:')}     {color(MAGENTA, f'#{repo.pr_number}')}"
        if repo.pr_title:
            pr_line += f" {color(DIM, '—')} {repo.pr_title}"
        body.append(pr_line)

    body.append(_format_worker_thread_line(repo))

    body.extend(_format_webhook_lines(repo))

    return body


def _format_worker_thread_line(repo: RepoStatus) -> str:
    """Worker-thread state line, background-highlighted whenever the worker
    is assigned to a task.

    The previous rule ("highlight iff the worker currently owns the
    session lock") flickered to off during the Python-only gaps between
    turns — the worker finishes a ``session.prompt`` (talker
    unregisters), processes the result (git ops, task bookkeeping), then
    calls ``session.prompt`` again.  Those gaps are still active work but
    read as idle in the old semantics.

    Use ``current_task`` as the stable \"worker is on a task\" signal —
    set whenever the engine is inside ``execute_task``, cleared only when
    the worker is napping / picking up / paused.  OR it with the classic
    talker check so webhook-preemption snapshots (where current_task may
    briefly be None during task rescopes) still read as busy when the
    worker lock is held.
    """
    state = _worker_thread_state(repo)
    is_active = repo.current_task is not None or _worker_is_agent_talker(repo)
    # NO_COLOR users need an alternate signal to the GREEN_BG highlight;
    # a leading "* " is visible when color is disabled.  Under color mode
    # GREEN_BG already provides the highlight, so the asterisk is omitted.
    # Inactive rows always get two spaces so the label column stays aligned.
    marker = "* " if (is_active and not color_enabled()) else "  "
    label = color(GREEN_BG, "Worker:") if is_active else color(BOLD, "Worker:")
    line = f"{marker}{label} {state}"
    return line


def _worker_thread_state(repo: RepoStatus) -> str:
    """Compact string describing what the worker thread itself is doing.

    Prefers the richest descriptor available: current task (with position
    and title) > PR-but-no-task > the live ``worker_what`` field >
    ``waiting for work``.
    Never shows webhook-thread activity — that's surfaced separately.
    """
    provider_status = repo.provider_status
    if provider_status is not None and provider_status.paused:
        until = (
            f" until {_format_reset_at(provider_status.resets_at)}"
            if provider_status.resets_at is not None
            else ""
        )
        return color(
            DARK_GRAY,
            f"paused for {provider_status.provider} reset{until}",
        )
    if repo.task_number is not None and repo.task_total is not None:
        uncertainty = "?" if repo.rescoping else ""
        suffix = f" — {repo.current_task}" if repo.current_task else ""
        return f"{color(BOLD, f'task {repo.task_number}/{repo.task_total}{uncertainty}')}{suffix}"
    if repo.current_task is not None:
        return f"task: {repo.current_task}"
    what = (repo.worker_what or "").strip()
    if what:
        return what
    return "waiting for work"


def _format_webhook_lines(repo: RepoStatus) -> list[str]:
    """Render up to :data:`_WEBHOOK_DISPLAY_CAP` webhook lines with the
    talker sorted to the top.  Returns ``[]`` when there are no webhooks.
    """
    webhooks = list(repo.webhook_activities)
    if not webhooks:
        return []
    talker = repo.claude_talker
    talker_tid = (
        talker.thread_id if talker is not None and talker.kind == "webhook" else None
    )
    # Stable sort: webhooks driving the agent first, then everything else in
    # original order.
    webhooks.sort(key=lambda w: 0 if w.thread_id == talker_tid else 1)

    shown = webhooks[:_WEBHOOK_DISPLAY_CAP]
    overflow = len(webhooks) - len(shown)
    lines: list[str] = []
    for i, w in enumerate(shown):
        branch = color(DIM, "└─" if overflow == 0 and i == len(shown) - 1 else "├─")
        is_talker = talker_tid is not None and w.thread_id == talker_tid
        wh_label = (
            color(YELLOW_BG, "webhook:") if is_talker else color(BOLD, "webhook:")
        )
        elapsed = color(DIM, f"({_format_uptime(w.elapsed_seconds)})")
        line = f"  {branch} {wh_label} {w.description} {elapsed}"
        lines.append(line)
    if overflow > 0:
        lines.append(
            color(DIM, f"  └─ +{overflow} more webhook{'s' if overflow != 1 else ''}")
        )
    return lines


def format_status(status: KennelStatus) -> str:
    """Format a KennelStatus as a human-readable string."""
    lines: list[str] = []

    if status.kennel_pid is not None:
        uptime_str = (
            f", uptime {_format_uptime(status.kennel_uptime)}"
            if status.kennel_uptime is not None
            else ""
        )
        lines.append(
            f"{color(BOLD, 'kennel:')} {color(GREEN, 'UP')} "
            f"{color(DIM, f'(pid {status.kennel_pid}{uptime_str})')}"
        )
    else:
        lines.append(f"{color(BOLD, 'kennel:')} {color(RED_BOLD, 'DOWN')}")

    provider_summary = _format_provider_summary_line(status.provider_statuses)
    if provider_summary is not None:
        lines.append(provider_summary)

    rate_limit_line = _format_rate_limit_line(status.rate_limit)
    if rate_limit_line is not None:
        lines.append(rate_limit_line)

    for repo in status.repos:
        section = [_format_repo_header(repo), *_format_repo_body(repo)]
        palette = palette_for(repo.provider)
        if palette is not None:
            bg = rgb_bg(*palette.dim_bg)
            section = [wrap_bg_line(bg, line) for line in section]
        lines.extend(section)

    return "\n".join(lines)
