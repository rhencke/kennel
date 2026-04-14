"""kennel status — reads per-repo fido state and formats a one-shot summary."""

from __future__ import annotations

import fcntl
import json
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kennel.config import RepoConfig
from kennel.state import State


@dataclass
class WebhookActivityInfo:
    """Lightweight in-flight webhook handler, used purely for display."""

    description: str
    elapsed_seconds: int


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
    # Newer fields default so callers (tests) can omit them safely.
    issue_title: str | None = None
    issue_elapsed_seconds: int | None = None
    pr_number: int | None = None
    pr_title: str | None = None
    task_number: int | None = None
    task_total: int | None = None
    worker_uptime: int | None = None
    webhook_activities: list[WebhookActivityInfo] = field(default_factory=list)
    session_owner: str | None = None
    session_alive: bool = False


@dataclass
class KennelStatus:
    kennel_pid: int | None
    kennel_uptime: int | None  # seconds
    repos: list[RepoStatus]


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
        name, path_str = spec.split(":", 1)
        if "/" not in name:
            continue
        repos.append(RepoConfig(name=name, work_dir=Path(path_str).expanduser()))
    return repos


def _kennel_pid() -> int | None:
    """Return the PID of the running kennel server, or None."""
    pids = _pgrep("kennel --port")
    return pids[0] if pids else None


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
) -> dict[str, dict[str, Any]]:
    """Query GET /status, returning {repo_name: {what, crash_count, last_crash_error, is_stuck}}."""
    try:
        with _urlopen(f"http://localhost:{port}/status", timeout=2) as resp:
            data = json.loads(resp.read())
        return {
            item["repo_name"]: {
                "what": item["what"],
                "crash_count": item["crash_count"],
                "last_crash_error": item["last_crash_error"],
                "is_stuck": item.get("is_stuck", False),
                "worker_uptime_seconds": item.get("worker_uptime_seconds"),
                "webhook_activities": item.get("webhook_activities", []),
                "session_owner": item.get("session_owner"),
                "session_alive": item.get("session_alive", False),
            }
            for item in data
            if "repo_name" in item and "what" in item
        }
    except Exception:  # noqa: BLE001
        return {}


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


def _read_state(fido_dir: Path) -> dict[str, Any]:
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


def _read_tasks(fido_dir: Path) -> list[dict[str, Any]]:
    """Read tasks.json from fido_dir, returning [] if absent or unreadable."""
    path = fido_dir / "tasks.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return []


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
    """Return (task_number, total_non_completed) for display.

    task_number is the 1-indexed position of the first in_progress or
    pending task among all non-completed tasks; total is the count of
    non-completed tasks.  Returns (None, None) when there are none.
    """
    non_completed = [t for t in task_list if t["status"] != "completed"]
    if not non_completed:
        return (None, None)
    for idx, t in enumerate(non_completed, start=1):
        if t["status"] == "in_progress":
            return (idx, len(non_completed))
    return (1, len(non_completed))


def _elapsed_since_iso(iso: str | None, *, _now=None) -> int | None:
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
            webhook_activities=webhook_activities,
            session_owner=session_owner,
            session_alive=session_alive,
        )

    fido_dir = git_dir / "fido"
    running = _fido_running(fido_dir / "lock")

    state = State(fido_dir).load()
    issue = state.get("issue")
    issue_title = state.get("issue_title")
    issue_elapsed = _elapsed_since_iso(state.get("issue_started_at"))
    pr_number = state.get("pr_number")
    pr_title = state.get("pr_title")

    task_list = _read_tasks(fido_dir)
    pending = sum(1 for t in task_list if t["status"] == "pending")
    completed = sum(1 for t in task_list if t["status"] == "completed")

    current = _current_task(task_list)
    task_number, task_total = _task_position(task_list)

    claude_pid = _claude_pid(fido_dir)
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
        webhook_activities=webhook_activities,
        session_owner=session_owner,
        session_alive=session_alive,
    )


def collect() -> KennelStatus:
    """Collect the full kennel + per-repo status."""
    pid = _kennel_pid()
    uptime = _process_uptime_seconds(pid) if pid is not None else None
    repo_configs = _repos_from_pid(pid) if pid is not None else []

    activities: dict[str, str] = {}
    if pid is not None:
        port = _port_from_pid(pid)
        if port is not None:
            activities = _fetch_activities(port)

    repos = []
    for rc in repo_configs:
        info = activities.get(rc.name)
        webhook_list = []
        worker_uptime_val: int | None = None
        if info:
            for w in info.get("webhook_activities") or []:
                webhook_list.append(
                    WebhookActivityInfo(
                        description=w["description"],
                        elapsed_seconds=int(w["elapsed_seconds"]),
                    )
                )
            wu = info.get("worker_uptime_seconds")
            worker_uptime_val = int(wu) if wu is not None else None
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
            )
        )
    return KennelStatus(kennel_pid=pid, kennel_uptime=uptime, repos=repos)


_SILENT_DISPLAY_THRESHOLD: int = 300  # seconds of claude silence before we note it


def _format_repo_header(repo: RepoStatus) -> str:
    """Top line: `<name>: fido <state> — <compact stats>`.

    Stats are comma-separated and only shown when they matter right now:
    `crashes N` (skipped when 0), `up X` (worker thread uptime, always
    shown when known), `BUSY Xm` (no activity for ≥5m; since #444 this is
    informational only, not a restart signal).
    """
    state = "fido running" if repo.fido_running else "fido idle"
    stats: list[str] = []
    if repo.crash_count > 0:
        stats.append(f"crashes {repo.crash_count}")
    if repo.worker_uptime is not None:
        stats.append(f"up {_format_uptime(repo.worker_uptime)}")
    if repo.worker_stuck:
        # worker_uptime includes time on the current task; there isn't a
        # separate "silent" counter yet, so we just flag BUSY.
        stats.append("BUSY")
    if repo.crash_count > 0 and repo.last_crash_error:
        stats.append(f"last crash: {repo.last_crash_error}")

    header = f"{repo.name}: {state}"
    if stats:
        header += " — " + ", ".join(stats)
    return header


def _format_repo_body(repo: RepoStatus) -> list[str]:
    """Indented sub-lines: Issue / PR / Task / claude / webhooks."""
    body: list[str] = []

    if repo.issue is None:
        body.append("  no assigned issues")
        return body

    issue_line = f"  Issue:  #{repo.issue}"
    if repo.issue_title:
        issue_line += f" — {repo.issue_title}"
    if repo.issue_elapsed_seconds is not None:
        issue_line += f"  (elapsed {_format_uptime(repo.issue_elapsed_seconds)})"
    body.append(issue_line)

    if repo.pr_number is not None:
        pr_line = f"  PR:     #{repo.pr_number}"
        if repo.pr_title:
            pr_line += f" — {repo.pr_title}"
        body.append(pr_line)

    if repo.task_number is not None and repo.task_total is not None:
        task_line = f"  Task:   {repo.task_number}/{repo.task_total}"
        if repo.current_task:
            task_line += f" — {repo.current_task}"
        body.append(task_line)
    elif repo.current_task:
        body.append(f"  Task:   {repo.current_task}")

    if repo.worker_what and not (repo.current_task or repo.pr_number):
        # Only show the live activity when nothing more specific fits.
        body.append(f"  Doing:  {repo.worker_what}")

    if repo.claude_pid is not None:
        claude_str = f"  └─ claude pid {repo.claude_pid}"
        parts: list[str] = []
        if repo.claude_uptime is not None:
            parts.append(f"running {_format_uptime(repo.claude_uptime)}")
        if repo.session_owner is not None:
            parts.append(f"held by {repo.session_owner}")
        elif repo.session_alive:
            parts.append("session idle")
        if parts:
            claude_str += f" ({', '.join(parts)})"
        body.append(claude_str)
    elif repo.session_alive:
        body.append("  └─ session alive (no pgrep match)")

    for w in repo.webhook_activities:
        body.append(
            f"  └─ webhook: {w.description} ({_format_uptime(w.elapsed_seconds)})"
        )

    return body


def format_status(status: KennelStatus) -> str:
    """Format a KennelStatus as a human-readable string."""
    lines: list[str] = []

    if status.kennel_pid is not None:
        uptime_str = (
            f", uptime {_format_uptime(status.kennel_uptime)}"
            if status.kennel_uptime is not None
            else ""
        )
        lines.append(f"kennel: UP (pid {status.kennel_pid}{uptime_str})")
    else:
        lines.append("kennel: DOWN")

    for repo in status.repos:
        lines.append(_format_repo_header(repo))
        lines.extend(_format_repo_body(repo))

    return "\n".join(lines)
