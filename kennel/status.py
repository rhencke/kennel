"""kennel status — reads per-repo fido state and formats a one-shot summary."""

from __future__ import annotations

import fcntl
import json
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kennel.config import RepoConfig


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
    """Query GET /status, returning {repo_name: {what, crash_count, last_crash_error}}."""
    try:
        with _urlopen(f"http://localhost:{port}/status", timeout=2) as resp:
            data = json.loads(resp.read())
        return {
            item["repo_name"]: {
                "what": item["what"],
                "crash_count": item["crash_count"],
                "last_crash_error": item["last_crash_error"],
            }
            for item in data
            if "repo_name" in item and "what" in item
        }
    except Exception:  # noqa: BLE001
        return {}


def _claude_pid(fido_dir: Path) -> int | None:
    """Return the PID of the claude process using fido_dir/system, or None."""
    pids = _pgrep(str(fido_dir / "system"))
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


def repo_status(
    repo_config: RepoConfig,
    worker_what: str | None = None,
    crash_count: int = 0,
    last_crash_error: str | None = None,
) -> RepoStatus:
    """Collect status for a single repo."""
    git_dir = _git_dir(repo_config.work_dir)
    if git_dir is None:
        return RepoStatus(
            name=repo_config.name,
            fido_running=False,
            issue=None,
            pending=0,
            completed=0,
            current_task=None,
            claude_pid=None,
            claude_uptime=None,
            worker_what=worker_what,
            crash_count=crash_count,
            last_crash_error=last_crash_error,
        )

    fido_dir = git_dir / "fido"
    running = _fido_running(fido_dir / "lock")

    state = _read_state(fido_dir)
    issue = state.get("issue")

    task_list = _read_tasks(fido_dir)
    pending = sum(1 for t in task_list if t["status"] == "pending")
    completed = sum(1 for t in task_list if t["status"] == "completed")

    current = _current_task(task_list)

    claude_pid = _claude_pid(fido_dir)
    claude_uptime = (
        _process_uptime_seconds(claude_pid) if claude_pid is not None else None
    )

    return RepoStatus(
        name=repo_config.name,
        fido_running=running,
        issue=issue,
        pending=pending,
        completed=completed,
        current_task=current,
        claude_pid=claude_pid,
        claude_uptime=claude_uptime,
        worker_what=worker_what,
        crash_count=crash_count,
        last_crash_error=last_crash_error,
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
        repos.append(
            repo_status(
                rc,
                worker_what=info["what"] if info else None,
                crash_count=info["crash_count"] if info else 0,
                last_crash_error=info["last_crash_error"] if info else None,
            )
        )
    return KennelStatus(kennel_pid=pid, kennel_uptime=uptime, repos=repos)


def format_status(status: KennelStatus) -> str:
    """Format a KennelStatus as a human-readable string."""
    lines = []

    # Kennel server line
    if status.kennel_pid is not None:
        uptime_str = (
            f", uptime {_format_uptime(status.kennel_uptime)}"
            if status.kennel_uptime is not None
            else ""
        )
        lines.append(f"kennel: UP (pid {status.kennel_pid}{uptime_str})")
    else:
        lines.append("kennel: DOWN")

    # Per-repo lines
    for repo in status.repos:
        parts = []

        if repo.fido_running:
            parts.append("fido running")
        else:
            parts.append("fido idle")

        if repo.worker_what:
            parts.append(repo.worker_what)

        if repo.issue is not None:
            total = repo.pending + repo.completed
            issue_str = f"issue #{repo.issue}"
            if repo.current_task:
                done = repo.completed
                issue_str += f', task {done + 1}/{total} "{repo.current_task}"'
            elif total > 0:
                issue_str += f", {repo.pending} pending"
            parts.append(issue_str)
        else:
            parts.append("no assigned issues")

        if repo.claude_pid is not None:
            claude_str = f"claude pid {repo.claude_pid}"
            if repo.claude_uptime is not None:
                claude_str += f" (running {_format_uptime(repo.claude_uptime)})"
            parts.append(claude_str)

        if repo.crash_count > 0:
            crash_str = f"crashed {repo.crash_count}x"
            if repo.last_crash_error:
                crash_str += f": {repo.last_crash_error}"
            parts.append(crash_str)

        lines.append(f"{repo.name}: {' — '.join(parts)}")

    return "\n".join(lines)
