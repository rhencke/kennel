"""Tests for kennel.watchdog — Watchdog class and run() entry point."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from kennel.config import RepoConfig
from kennel.watchdog import (
    _MAX_STALE_COUNT,  # noqa: PLC2701
    _STALE_THRESHOLD,  # noqa: PLC2701
    Watchdog,
    run,
)


def _repo(name: str = "owner/repo") -> RepoConfig:
    return RepoConfig(name=name, work_dir=Path("/tmp/repo"))


def _make(repos: dict[str, RepoConfig] | None = None) -> tuple[Watchdog, MagicMock]:
    if repos is None:
        repos = {"owner/repo": _repo()}
    registry = MagicMock()
    registry.is_stale.return_value = False
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
        registry.get_thread_crash_error.return_value = None
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
        registry.get_thread_crash_error.return_value = None
        w.run()
        registry.start.assert_called_once_with(dead_cfg)

    def test_restarts_multiple_dead_threads(self) -> None:
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        repos = {"org/a": repo_a, "org/b": repo_b}
        w, registry = _make(repos)
        registry.is_alive.return_value = False
        registry.get_thread_crash_error.return_value = None
        w.run()
        assert registry.start.call_count == 2
        registry.start.assert_any_call(repo_a)
        registry.start.assert_any_call(repo_b)

    def test_records_crash_before_restart_when_crash_error_set(self) -> None:
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive.return_value = False
        registry.get_thread_crash_error.return_value = "RuntimeError: boom"
        w.run()
        registry.record_crash.assert_called_once_with(
            "owner/repo", "RuntimeError: boom"
        )
        registry.start.assert_called_once_with(repo_cfg)

    def test_record_crash_called_before_start(self) -> None:
        call_order: list[str] = []
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive.return_value = False
        registry.get_thread_crash_error.return_value = "ValueError: oops"
        registry.record_crash.side_effect = lambda *_: call_order.append("record_crash")
        registry.start.side_effect = lambda *_: call_order.append("start")
        w.run()
        assert call_order == ["record_crash", "start"]

    def test_does_not_record_crash_when_crash_error_is_none(self) -> None:
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive.return_value = False
        registry.get_thread_crash_error.return_value = None
        w.run()
        registry.record_crash.assert_not_called()
        registry.start.assert_called_once_with(repo_cfg)

    def test_no_repos_is_no_op(self) -> None:
        w, registry = _make({})
        w.run()
        registry.is_alive.assert_not_called()
        registry.start.assert_not_called()


# ── Watchdog stale detection ───────────────────────────────────────────────────


class TestWatchdogStale:
    def _stale_make(self, max_stale: int = 2) -> tuple[Watchdog, MagicMock]:
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = True
        registry.is_stale.return_value = True
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=max_stale,
        )
        return w, registry

    def test_warns_on_first_stale_detection(self) -> None:
        w, _ = self._stale_make()
        from unittest.mock import patch

        with patch("kennel.watchdog.log") as mock_log:
            w.run()
        mock_log.warning.assert_called_once()
        args = mock_log.warning.call_args.args
        assert "owner/repo" in args[1]

    def test_does_not_restart_on_first_stale_detection(self) -> None:
        w, registry = self._stale_make(max_stale=2)
        w.run()
        registry.start.assert_not_called()
        registry.stop_and_join.assert_not_called()

    def test_increments_stale_count_each_detection(self) -> None:
        w, _ = self._stale_make(max_stale=5)
        w.run()
        assert w._stale_counts["owner/repo"] == 1
        w.run()
        assert w._stale_counts["owner/repo"] == 2

    def test_forces_restart_at_max_stale_count(self) -> None:
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = True
        registry.is_stale.return_value = True
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=2,
        )
        w.run()  # count → 1, no restart
        registry.start.assert_not_called()
        w.run()  # count → 2 == max, restart
        registry.stop_and_join.assert_called_once_with("owner/repo")
        registry.start.assert_called_once_with(repo_cfg)

    def test_stop_and_join_called_before_start_on_stale_restart(self) -> None:
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = True
        registry.is_stale.return_value = True
        call_order: list[str] = []
        registry.stop_and_join.side_effect = lambda *_: call_order.append(
            "stop_and_join"
        )
        registry.start.side_effect = lambda *_: call_order.append("start")
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=1,
        )
        w.run()
        assert call_order == ["stop_and_join", "start"]

    def test_records_crash_before_stale_restart(self) -> None:
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = True
        registry.is_stale.return_value = True
        call_order: list[str] = []
        registry.record_crash.side_effect = lambda *_: call_order.append("record_crash")
        registry.stop_and_join.side_effect = lambda *_: call_order.append(
            "stop_and_join"
        )
        registry.start.side_effect = lambda *_: call_order.append("start")
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=1,
        )
        w.run()
        assert call_order == ["record_crash", "stop_and_join", "start"]

    def test_stale_restart_records_crash_with_stuck_message(self) -> None:
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = True
        registry.is_stale.return_value = True
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=1,
        )
        w.run()
        registry.record_crash.assert_called_once()
        _, error = registry.record_crash.call_args.args
        assert "stuck" in error

    def test_stale_count_resets_after_forced_restart(self) -> None:
        w, _ = self._stale_make(max_stale=1)
        w.run()  # triggers restart, resets count
        assert "owner/repo" not in w._stale_counts

    def test_stale_count_resets_on_healthy_check(self) -> None:
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = True
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=5,
        )
        registry.is_stale.return_value = True
        w.run()
        w.run()
        assert w._stale_counts.get("owner/repo", 0) == 2
        registry.is_stale.return_value = False
        w.run()  # healthy — resets
        assert "owner/repo" not in w._stale_counts

    def test_stale_count_resets_on_dead_thread_restart(self) -> None:
        repo_cfg = _repo()
        registry = MagicMock()
        w = Watchdog(
            registry,
            {"owner/repo": repo_cfg},
            _stale_threshold=300.0,
            _max_stale_count=5,
        )
        # Build up a stale count
        registry.is_alive.return_value = True
        registry.is_stale.return_value = True
        w.run()
        assert w._stale_counts.get("owner/repo", 0) == 1
        # Thread dies next iteration
        registry.is_alive.return_value = False
        registry.get_thread_crash_error.return_value = None
        w.run()
        assert "owner/repo" not in w._stale_counts

    def test_default_stale_threshold_constant(self) -> None:
        assert _STALE_THRESHOLD == 600.0

    def test_default_max_stale_count_constant(self) -> None:
        assert _MAX_STALE_COUNT == 2

    def test_watchdog_accepts_stale_threshold_kwarg(self) -> None:
        w = Watchdog(MagicMock(), {}, _stale_threshold=120.0)
        assert w._stale_threshold == 120.0

    def test_watchdog_accepts_max_stale_count_kwarg(self) -> None:
        w = Watchdog(MagicMock(), {}, _max_stale_count=3)
        assert w._max_stale_count == 3

    def test_is_stale_called_with_threshold(self) -> None:
        w, registry = self._stale_make()
        registry.is_stale.return_value = False
        w.run()
        registry.is_stale.assert_called_once_with("owner/repo", 300.0)

    def test_stale_check_skipped_when_thread_dead(self) -> None:
        """is_stale must not be called for a dead thread."""
        repo_cfg = _repo()
        registry = MagicMock()
        registry.is_alive.return_value = False
        registry.get_thread_crash_error.return_value = None
        w = Watchdog(registry, {"owner/repo": repo_cfg})
        w.run()
        registry.is_stale.assert_not_called()


# ── module-level run() ─────────────────────────────────────────────────────────


def _registry(*, alive: bool = True, stale: bool = False) -> MagicMock:
    """Return a mock registry with is_alive and is_stale pre-configured."""
    reg = MagicMock()
    reg.is_alive.return_value = alive
    reg.is_stale.return_value = stale
    return reg


class TestStartThread:
    def _repos(self, tmp_path: Path) -> dict:
        return {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}

    def test_returns_daemon_thread(self, tmp_path: Path) -> None:
        t = Watchdog(_registry(), self._repos(tmp_path)).start_thread(_interval=60.0)
        assert t.daemon

    def test_thread_name_is_watchdog(self, tmp_path: Path) -> None:
        t = Watchdog(_registry(), self._repos(tmp_path)).start_thread(_interval=60.0)
        assert t.name == "watchdog"

    def test_thread_is_alive(self, tmp_path: Path) -> None:
        t = Watchdog(_registry(), self._repos(tmp_path)).start_thread(_interval=60.0)
        assert t.is_alive()

    def test_calls_run_periodically(self, tmp_path: Path) -> None:
        reg = _registry()
        Watchdog(reg, self._repos(tmp_path)).start_thread(_interval=0.01)
        time.sleep(0.1)
        reg.is_alive.assert_called()

    def test_restarts_dead_worker(self, tmp_path: Path) -> None:
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        reg = _registry(alive=False)
        Watchdog(reg, {"owner/repo": repo_cfg}).start_thread(_interval=0.01)
        time.sleep(0.1)
        reg.start.assert_called_with(repo_cfg)


class TestModuleLevelRun:
    def test_delegates_to_watchdog(self) -> None:
        repo_cfg = _repo()
        repos = {"owner/repo": repo_cfg}
        reg = _registry()
        result = run(reg, repos)
        assert result == 0
        reg.is_alive.assert_called_once_with("owner/repo")

    def test_returns_zero(self) -> None:
        assert run(_registry(), {"owner/repo": _repo()}) == 0
