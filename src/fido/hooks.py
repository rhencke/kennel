"""settings.local.json hook management for fido workers."""

import json
from pathlib import Path
from typing import Any


def _settings_path(work_dir: Path) -> Path:
    return work_dir / ".claude" / "settings.local.json"


def _exclude_path(work_dir: Path) -> Path:
    """Return .git/info/exclude for the repo at work_dir."""
    return work_dir / ".git" / "info" / "exclude"


def ensure_gitexcluded(work_dir: Path) -> None:
    """Add .claude/settings.local.json to .git/info/exclude if not already there."""
    exclude = _exclude_path(work_dir)
    exclude.parent.mkdir(parents=True, exist_ok=True)
    entry = ".claude/settings.local.json"
    if exclude.exists():
        lines = exclude.read_text().splitlines()
        if entry in lines:
            return
    with exclude.open("a") as f:
        f.write(entry + "\n")


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError, OSError:
            return {}
    return {}


def _save(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))


def _hook_entry(command: str, matcher: str = "") -> dict[str, Any]:
    return {"matcher": matcher, "hooks": [{"type": "command", "command": command}]}


def add_hooks(
    work_dir: Path,
    compact_command: str,
    sync_command: str,
    sync_tools: tuple[str, ...] = (
        "TaskCreate",
        "TaskUpdate",
        "TaskDelete",
        "TodoWrite",
        "TodoRead",
    ),
) -> None:
    """Add PostCompact and PostToolUse hooks to settings.local.json.

    compact_command: shell command to run on PostCompact.
    sync_command: shell command to run on PostToolUse for task-mutation tools.
    sync_tools: tool names that trigger the sync hook.
    """
    path = _settings_path(work_dir)
    cfg = _load(path)

    # PostCompact hook
    cfg.setdefault("hooks", {}).setdefault("PostCompact", [])
    compact_entry = _hook_entry(compact_command)
    if compact_entry not in cfg["hooks"]["PostCompact"]:
        cfg["hooks"]["PostCompact"].append(compact_entry)

    # PostToolUse hooks for task-mutation tools
    cfg["hooks"].setdefault("PostToolUse", [])
    for tool in sync_tools:
        tool_entry = _hook_entry(sync_command, matcher=tool)
        if tool_entry not in cfg["hooks"]["PostToolUse"]:
            cfg["hooks"]["PostToolUse"].append(tool_entry)

    _save(path, cfg)


def remove_hooks(
    work_dir: Path,
    compact_command: str,
    sync_command: str,
) -> None:
    """Remove PostCompact and PostToolUse hooks added by add_hooks.

    Cleans up empty hook lists and the hooks key itself if empty.
    """
    path = _settings_path(work_dir)
    if not path.exists():
        return
    cfg = _load(path)

    hooks = cfg.get("hooks", {})

    # Remove PostCompact entry matching compact_command
    post_compact = hooks.get("PostCompact", [])
    hooks["PostCompact"] = [
        h
        for h in post_compact
        if not any(e.get("command") == compact_command for e in h.get("hooks", []))
    ]
    if not hooks["PostCompact"]:
        del hooks["PostCompact"]

    # Remove PostToolUse entries matching sync_command
    post_tool = hooks.get("PostToolUse", [])
    hooks["PostToolUse"] = [
        h
        for h in post_tool
        if not any(e.get("command") == sync_command for e in h.get("hooks", []))
    ]
    if not hooks["PostToolUse"]:
        del hooks["PostToolUse"]

    if not hooks:
        cfg.pop("hooks", None)
    else:
        cfg["hooks"] = hooks

    _save(path, cfg)
