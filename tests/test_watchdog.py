"""Tests for fido.watchdog — Watchdog class and run() entry point."""

import time
from pathlib import Path
from unittest.mock import MagicMock

from fido.config import RepoConfig as _RepoConfig
from fido.provider import ProviderID
from fido.watchdog import (
    _RECONCILE_INTERVAL,  # noqa: PLC2701
    _STALE_THRESHOLD,  # noqa: PLC2701
    ReconcileWatchdog,
    Watchdog,
    run,
)


class RepoConfig(_RepoConfig):
    def __init__(
        self,
        *args: object,
        provider: ProviderID = ProviderID.CLAUDE_CODE,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, provider=provider, **kwargs)


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
        """A live thread is never restarted, even if it looks stale.  Stale
        threads are claude's problem — claude has its own idle timeout."""
        w, registry = _make()
        registry.is_alive.return_value = True
        w.run()
        registry.start.assert_not_called()
        registry.stop_and_join.assert_not_called()

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

    def test_is_stale_never_called_for_restart(self) -> None:
        """Stale detection is not a restart trigger.  is_stale may be called
        elsewhere (e.g. /status endpoint) but never by the watchdog itself."""
        w, registry = _make()
        registry.is_alive.return_value = True
        w.run()
        registry.is_stale.assert_not_called()

    def test_does_not_stop_and_join_alive_thread(self) -> None:
        w, registry = _make()
        registry.is_alive.return_value = True
        w.run()
        registry.stop_and_join.assert_not_called()


# ── display-only constants ────────────────────────────────────────────────────


class TestConstants:
    def test_stale_threshold_is_display_only(self) -> None:
        """_STALE_THRESHOLD exists for /status endpoint display.  It is not
        consumed by the Watchdog class itself — documented via this test."""
        assert _STALE_THRESHOLD > 0


# ── Watchdog.start_thread ─────────────────────────────────────────────────────


def _registry(*, alive: bool = True) -> MagicMock:
    reg = MagicMock()
    reg.is_alive.return_value = alive
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


# ── module-level run() ─────────────────────────────────────────────────────────


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


# ── ReconcileWatchdog (closes #812) ───────────────────────────────────────────


def _reconcile(
    repos: dict[str, RepoConfig] | None = None,
    *,
    cache_loaded: bool = True,
) -> tuple[ReconcileWatchdog, MagicMock, MagicMock, MagicMock]:
    if repos is None:
        repos = {"owner/repo": _repo()}
    registry = MagicMock()
    cache = MagicMock()
    cache.is_loaded = cache_loaded
    registry.get_issue_cache.return_value = cache
    gh = MagicMock()
    return ReconcileWatchdog(registry, repos, gh), registry, cache, gh


class TestReconcileWatchdogRun:
    def test_returns_zero(self) -> None:
        rw, _registry, _cache, gh = _reconcile(cache_loaded=False)
        assert rw.run() == 0
        gh.find_all_open_issues.assert_not_called()

    def test_skips_repo_when_cache_not_loaded(self) -> None:
        rw, _registry, cache, gh = _reconcile(cache_loaded=False)
        rw.run()
        gh.find_all_open_issues.assert_not_called()
        cache.reconcile_with_inventory.assert_not_called()

    def test_reconciles_loaded_cache(self) -> None:
        rw, _registry, cache, gh = _reconcile()
        gh.find_all_open_issues.return_value = [{"number": 1}]
        cache.reconcile_with_inventory.return_value = 0
        rw.run()
        gh.find_all_open_issues.assert_called_once_with("owner", "repo")
        cache.reconcile_with_inventory.assert_called_once()
        args, kwargs = cache.reconcile_with_inventory.call_args
        assert args[0] == [{"number": 1}]
        assert "snapshot_started_at" in kwargs

    def test_continues_to_next_repo_when_inventory_call_raises(self) -> None:
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        rw, _registry, cache, gh = _reconcile({"org/a": repo_a, "org/b": repo_b})

        def side_effect(owner: str, _name: str) -> list:
            if owner == "org" and _name == "a":
                raise RuntimeError("rate limited")
            return [{"number": 9}]

        gh.find_all_open_issues.side_effect = side_effect
        rw.run()
        # b succeeded even though a raised
        cache.reconcile_with_inventory.assert_called_once()

    def test_handles_multiple_repos_independently(self) -> None:
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        rw, _registry, cache, gh = _reconcile({"org/a": repo_a, "org/b": repo_b})
        gh.find_all_open_issues.return_value = []
        cache.reconcile_with_inventory.return_value = 0
        rw.run()
        assert gh.find_all_open_issues.call_count == 2


class TestReconcileWatchdogStartThread:
    def test_returns_daemon_thread(self) -> None:
        rw, _registry, _cache, _gh = _reconcile(cache_loaded=False)
        t = rw.start_thread(_interval=60.0)
        assert t.daemon

    def test_thread_name_is_reconcile_watchdog(self) -> None:
        rw, _registry, _cache, _gh = _reconcile(cache_loaded=False)
        t = rw.start_thread(_interval=60.0)
        assert t.name == "reconcile-watchdog"

    def test_thread_is_alive(self) -> None:
        rw, _registry, _cache, _gh = _reconcile(cache_loaded=False)
        t = rw.start_thread(_interval=60.0)
        assert t.is_alive()

    def test_calls_run_periodically(self) -> None:
        rw, registry, _cache, _gh = _reconcile(cache_loaded=False)
        rw.start_thread(_interval=0.01)
        time.sleep(0.1)
        registry.get_issue_cache.assert_called()


class TestReconcileInterval:
    def test_default_interval_is_one_hour(self) -> None:
        assert _RECONCILE_INTERVAL == 3600.0
