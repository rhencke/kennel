import json
from pathlib import Path

from fido.hooks import (
    _exclude_path,
    _hook_entry,
    _load,
    _save,
    _settings_path,
    add_hooks,
    ensure_gitexcluded,
    remove_hooks,
)


class TestHelpers:
    def test_settings_path(self, tmp_path: Path) -> None:
        assert _settings_path(tmp_path) == tmp_path / ".claude" / "settings.local.json"

    def test_exclude_path(self, tmp_path: Path) -> None:
        assert _exclude_path(tmp_path) == tmp_path / ".git" / "info" / "exclude"

    def test_hook_entry_defaults(self) -> None:
        entry = _hook_entry("bash foo.sh")
        assert entry == {
            "matcher": "",
            "hooks": [{"type": "command", "command": "bash foo.sh"}],
        }

    def test_hook_entry_with_matcher(self) -> None:
        entry = _hook_entry("bash foo.sh", matcher="TaskCreate")
        assert entry["matcher"] == "TaskCreate"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        result = _load(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        p.write_text('{"a": 1}')
        assert _load(p) == {"a": 1}

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        p.write_text("not json")
        assert _load(p) == {}

    def test_save_creates_parents(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "cfg.json"
        _save(p, {"x": 42})
        assert json.loads(p.read_text()) == {"x": 42}

    def test_save_indent_2(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        _save(p, {"a": 1})
        assert "  " in p.read_text()


class TestEnsureGitexcluded:
    def _make_git_info(self, tmp_path: Path) -> Path:
        info_dir = tmp_path / ".git" / "info"
        info_dir.mkdir(parents=True)
        return info_dir / "exclude"

    def test_adds_entry_when_missing(self, tmp_path: Path) -> None:
        self._make_git_info(tmp_path)
        ensure_gitexcluded(tmp_path)
        exclude = _exclude_path(tmp_path)
        assert ".claude/settings.local.json" in exclude.read_text().splitlines()

    def test_idempotent(self, tmp_path: Path) -> None:
        self._make_git_info(tmp_path)
        ensure_gitexcluded(tmp_path)
        ensure_gitexcluded(tmp_path)
        lines = _exclude_path(tmp_path).read_text().splitlines()
        assert lines.count(".claude/settings.local.json") == 1

    def test_does_not_duplicate_existing_entry(self, tmp_path: Path) -> None:
        exclude = self._make_git_info(tmp_path)
        exclude.write_text(".claude/settings.local.json\n")
        ensure_gitexcluded(tmp_path)
        lines = exclude.read_text().splitlines()
        assert lines.count(".claude/settings.local.json") == 1

    def test_creates_info_dir_if_missing(self, tmp_path: Path) -> None:
        # No .git/info dir
        (tmp_path / ".git").mkdir()
        ensure_gitexcluded(tmp_path)
        exclude = _exclude_path(tmp_path)
        assert exclude.exists()
        assert ".claude/settings.local.json" in exclude.read_text().splitlines()

    def test_appends_to_existing_content(self, tmp_path: Path) -> None:
        exclude = self._make_git_info(tmp_path)
        exclude.write_text("*.pyc\n")
        ensure_gitexcluded(tmp_path)
        lines = exclude.read_text().splitlines()
        assert "*.pyc" in lines
        assert ".claude/settings.local.json" in lines


class TestAddHooks:
    def test_creates_settings_file(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        path = _settings_path(tmp_path)
        assert path.exists()

    def test_adds_post_compact_hook(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        entries = cfg["hooks"]["PostCompact"]
        assert any(
            any(e.get("command") == "bash compact.sh" for e in h["hooks"])
            for h in entries
        )

    def test_adds_post_tool_use_hooks(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        entries = cfg["hooks"]["PostToolUse"]
        commands = [e["command"] for h in entries for e in h["hooks"]]
        assert all(c == "bash sync.sh" for c in commands)

    def test_default_sync_tools(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        matchers = {h["matcher"] for h in cfg["hooks"]["PostToolUse"]}
        assert matchers == {
            "TaskCreate",
            "TaskUpdate",
            "TaskDelete",
            "TodoWrite",
            "TodoRead",
        }

    def test_custom_sync_tools(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "c", "s", sync_tools=("ToolA", "ToolB"))
        cfg = json.loads(_settings_path(tmp_path).read_text())
        matchers = {h["matcher"] for h in cfg["hooks"]["PostToolUse"]}
        assert matchers == {"ToolA", "ToolB"}

    def test_idempotent_compact(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        assert len(cfg["hooks"]["PostCompact"]) == 1

    def test_idempotent_tool_use(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        # 5 default tools, each once
        assert len(cfg["hooks"]["PostToolUse"]) == 5

    def test_preserves_existing_config(self, tmp_path: Path) -> None:
        path = _settings_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": "claude-opus"}))
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(path.read_text())
        assert cfg["model"] == "claude-opus"
        assert "hooks" in cfg

    def test_does_not_duplicate_compact_when_called_twice(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "c1", "s1")
        add_hooks(tmp_path, "c2", "s2")  # different commands
        cfg = json.loads(_settings_path(tmp_path).read_text())
        assert len(cfg["hooks"]["PostCompact"]) == 2


class TestRemoveHooks:
    def test_removes_compact_hook(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        assert "hooks" not in cfg

    def test_removes_tool_use_hooks(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        assert "PostToolUse" not in cfg.get("hooks", {})

    def test_noop_when_file_missing(self, tmp_path: Path) -> None:
        # Should not raise
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")

    def test_preserves_other_hooks(self, tmp_path: Path) -> None:
        path = _settings_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "hooks": {
                "PostCompact": [
                    {"matcher": "", "hooks": [{"type": "command", "command": "other"}]},
                    {
                        "matcher": "",
                        "hooks": [{"type": "command", "command": "bash compact.sh"}],
                    },
                ]
            }
        }
        path.write_text(json.dumps(cfg))
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        result = json.loads(path.read_text())
        remaining = result["hooks"]["PostCompact"]
        assert len(remaining) == 1
        assert remaining[0]["hooks"][0]["command"] == "other"

    def test_preserves_other_tool_use_hooks(self, tmp_path: Path) -> None:
        path = _settings_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "X",
                        "hooks": [{"type": "command", "command": "other"}],
                    },
                    {
                        "matcher": "Y",
                        "hooks": [{"type": "command", "command": "bash sync.sh"}],
                    },
                ]
            }
        }
        path.write_text(json.dumps(cfg))
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        result = json.loads(path.read_text())
        remaining = result["hooks"]["PostToolUse"]
        assert len(remaining) == 1
        assert remaining[0]["hooks"][0]["command"] == "other"

    def test_cleans_up_empty_hooks_key(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        assert "hooks" not in cfg

    def test_idempotent(self, tmp_path: Path) -> None:
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(_settings_path(tmp_path).read_text())
        assert "hooks" not in cfg

    def test_preserves_other_top_level_config(self, tmp_path: Path) -> None:
        path = _settings_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": "sonnet"}))
        add_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        remove_hooks(tmp_path, "bash compact.sh", "bash sync.sh")
        cfg = json.loads(path.read_text())
        assert cfg.get("model") == "sonnet"
        assert "hooks" not in cfg
