"""Tests for kennel.watchdog — Watchdog class and run() entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from kennel.config import RepoConfig
from kennel.watchdog import Watchdog, run


def _repo(name: str = "owner/repo") -> RepoConfig:
    return RepoConfig(name=name, work_dir=Path("/tmp/repo"))


def _make(repos: dict[str, RepoConfig] | None = None) -> tuple[Watchdog, MagicMock]:
    if repos is None:
        repos = {"owner/repo": _repo()}
    registry = MagicMock()
    return Watchdog(registry, repos), registry


# ── Watchdog.run ───────────────────────────────────────────────────────────────


class TestWatchdogRun:
    def test_returns_zero(self) -> None:
        w, registry = _make()
        registry.is_alive.return_value = True
        assert w.run() == 0

    def test_does_nothing_when_thread_alive(self) -> None:
        w, registry = _make()
        registry.is_alive.return_value = True
        w.run()
        registry.start.assert_not_called()

    def test_restarts_dead_thread(self) -> None:
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive.return_value = False
        w.run()
        registry.start.assert_called_once_with(repo_cfg)

    def test_checks_is_alive_with_repo_name(self) -> None:
        repo_cfg = _repo("myorg/myrepo")
        w, registry = _make({"myorg/myrepo": repo_cfg})
        registry.is_alive.return_value = True
        w.run()
        registry.is_alive.assert_called_once_with("myorg/myrepo")

    def test_restarts_only_dead_threads_across_multiple_repos(self) -> None:
        alive_cfg = _repo("org/alive")
        dead_cfg = _repo("org/dead")
        repos = {"org/alive": alive_cfg, "org/dead": dead_cfg}
        w, registry = _make(repos)

        def is_alive(name: str) -> bool:
            return name == "org/alive"

        registry.is_alive.side_effect = is_alive
        w.run()
        registry.start.assert_called_once_with(dead_cfg)

    def test_restarts_multiple_dead_threads(self) -> None:
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        repos = {"org/a": repo_a, "org/b": repo_b}
        w, registry = _make(repos)
        registry.is_alive.return_value = False
        w.run()
        assert registry.start.call_count == 2
        registry.start.assert_any_call(repo_a)
        registry.start.assert_any_call(repo_b)

    def test_no_repos_is_no_op(self) -> None:
        w, registry = _make({})
        w.run()
        registry.is_alive.assert_not_called()
        registry.start.assert_not_called()


# ── module-level run() ─────────────────────────────────────────────────────────


class TestModuleLevelRun:
    def test_delegates_to_watchdog(self) -> None:
        repo_cfg = _repo()
        repos = {"owner/repo": repo_cfg}
        registry = MagicMock()
        registry.is_alive.return_value = True
        result = run(registry, repos)
        assert result == 0
        registry.is_alive.assert_called_once_with("owner/repo")

    def test_returns_zero(self) -> None:
        registry = MagicMock()
        registry.is_alive.return_value = True
        assert run(registry, {"owner/repo": _repo()}) == 0
