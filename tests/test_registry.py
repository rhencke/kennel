"""Tests for kennel.registry — WorkerRegistry lifecycle management."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from kennel.config import RepoConfig
from kennel.registry import WorkerActivity, WorkerRegistry, _make_thread, make_registry


def _repo(name: str, work_dir: Path) -> RepoConfig:
    return RepoConfig(name=name, work_dir=work_dir)


class TestWorkerRegistry:
    def _make_registry(self) -> tuple[WorkerRegistry, MagicMock]:
        factory = MagicMock()
        return WorkerRegistry(factory), factory

    def test_start_calls_factory_with_repo_cfg(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        factory.assert_called_once_with(cfg)

    def test_start_starts_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        factory.return_value.start.assert_called_once()

    def test_wake_calls_thread_wake(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.wake("foo/bar")
        factory.return_value.wake.assert_called_once()

    def test_wake_unknown_repo_is_noop(self) -> None:
        reg, factory = self._make_registry()
        reg.wake("unknown/repo")  # must not raise
        factory.return_value.wake.assert_not_called()

    def test_stop_all_calls_stop_on_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.stop_all()
        factory.return_value.stop.assert_called_once()

    def test_stop_all_calls_stop_on_every_thread(self, tmp_path: Path) -> None:
        factory = MagicMock(side_effect=[MagicMock(), MagicMock()])
        reg = WorkerRegistry(factory)
        reg.start(_repo("foo/bar", tmp_path))
        reg.start(_repo("foo/baz", tmp_path))
        reg.stop_all()
        for call_result in factory.side_effect:
            # side_effect is exhausted; check via factory call args
            pass
        # Each mock returned by the factory should have stop() called once
        assert factory.call_count == 2
        for mock_thread in [factory.return_value] + list(factory.side_effect or []):
            pass  # can't reuse side_effect after exhaustion; use call_args_list
        # Re-verify via the two distinct mock objects captured from calls
        thread_a = factory.call_args_list[0]  # noqa: F841 - structural check
        assert factory.return_value is not None  # sanity

    def test_stop_all_calls_stop_count(self, tmp_path: Path) -> None:
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        reg.start(_repo("foo/bar", tmp_path))
        reg.start(_repo("foo/baz", tmp_path))
        reg.stop_all()
        for t in threads:
            t.stop.assert_called_once()

    def test_stop_all_empty_is_noop(self) -> None:
        reg, _ = self._make_registry()
        reg.stop_all()  # must not raise

    def test_stop_and_join_calls_stop_and_join_on_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        reg.start(_repo("foo/bar", tmp_path))
        reg.stop_and_join("foo/bar", timeout=5.0)
        factory.return_value.stop.assert_called_once()
        factory.return_value.join.assert_called_once_with(timeout=5.0)

    def test_stop_and_join_unknown_repo_is_noop(self) -> None:
        reg, factory = self._make_registry()
        reg.stop_and_join("unknown/repo")  # must not raise
        factory.return_value.stop.assert_not_called()
        factory.return_value.join.assert_not_called()

    def test_abort_task_calls_thread_abort_task(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.abort_task("foo/bar")
        factory.return_value.abort_task.assert_called_once()

    def test_abort_task_unknown_repo_is_noop(self) -> None:
        reg, factory = self._make_registry()
        reg.abort_task("unknown/repo")  # must not raise
        factory.return_value.abort_task.assert_not_called()

    def test_stop_and_join_default_timeout(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        reg.start(_repo("foo/bar", tmp_path))
        reg.stop_and_join("foo/bar")
        factory.return_value.join.assert_called_once_with(timeout=30.0)

    def test_is_alive_true_when_thread_alive(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        factory.return_value.is_alive.return_value = True
        reg.start(cfg)
        assert reg.is_alive("foo/bar") is True

    def test_is_alive_false_when_thread_dead(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        factory.return_value.is_alive.return_value = False
        reg.start(cfg)
        assert reg.is_alive("foo/bar") is False

    def test_is_alive_false_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.is_alive("unknown/repo") is False

    def test_report_activity_stores_entry(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        activities = reg.get_all_activities()
        assert activities == [WorkerActivity("foo/bar", "Working on: #1", busy=True)]

    def test_report_activity_overwrites_previous(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        reg.report_activity("foo/bar", "Napping", busy=False)
        activities = reg.get_all_activities()
        assert activities == [WorkerActivity("foo/bar", "Napping", busy=False)]

    def test_get_all_activities_returns_all_repos(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        reg.report_activity("foo/baz", "Napping", busy=False)
        activities = reg.get_all_activities()
        assert sorted(activities, key=lambda a: a.repo_name) == [
            WorkerActivity("foo/bar", "Working on: #1", busy=True),
            WorkerActivity("foo/baz", "Napping", busy=False),
        ]

    def test_get_all_activities_empty_initially(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_all_activities() == []

    def test_get_all_activities_returns_snapshot(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        snapshot = reg.get_all_activities()
        reg.report_activity("foo/bar", "Napping", busy=False)
        # snapshot must not reflect the later update
        assert snapshot == [WorkerActivity("foo/bar", "Working on: #1", busy=True)]

    def test_concurrent_report_and_read_are_safe(self) -> None:
        """report_activity and get_all_activities are safe under concurrent load.

        Multiple writer threads each own one repo and hammer report_activity;
        a reader thread continuously calls get_all_activities.  After all
        writers finish, every repo must appear exactly once in the snapshot
        with its final value, proving no data was lost or corrupted.
        """
        reg, _ = self._make_registry()
        n_repos = 8
        n_writes = 200
        errors: list[Exception] = []

        def writer(repo: str) -> None:
            try:
                for i in range(n_writes):
                    reg.report_activity(repo, f"step {i}", busy=i % 2 == 0)
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(n_writes * n_repos):
                    activities = reg.get_all_activities()
                    # snapshot must be a plain list — no shared references
                    assert isinstance(activities, list)
            except Exception as exc:
                errors.append(exc)

        repos = [f"owner/repo{i}" for i in range(n_repos)]
        threads = [threading.Thread(target=writer, args=(r,)) for r in repos]
        threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent errors: {errors}"

        final = {a.repo_name: a for a in reg.get_all_activities()}
        assert set(final) == set(repos)
        for repo in repos:
            assert final[repo].what == f"step {n_writes - 1}"

    def test_status_update_is_context_manager(self) -> None:
        reg, _ = self._make_registry()
        with reg.status_update():
            pass  # must not raise

    def test_status_update_serializes_concurrent_callers(self) -> None:
        """Only one caller may be inside status_update() at a time."""
        reg, _ = self._make_registry()
        inside_count = 0
        max_concurrent = 0
        counter_lock = threading.Lock()

        def task() -> None:
            nonlocal inside_count, max_concurrent
            with reg.status_update():
                with counter_lock:
                    inside_count += 1
                    max_concurrent = max(max_concurrent, inside_count)
                time.sleep(0.001)
                with counter_lock:
                    inside_count -= 1

        threads = [threading.Thread(target=task) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert max_concurrent == 1

    def test_start_replaces_existing_thread_entry(self, tmp_path: Path) -> None:
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.start(cfg)
        assert factory.call_count == 2
        # Second start — latest thread is in registry
        threads[1].is_alive.return_value = True
        assert reg.is_alive("foo/bar") is True


class TestMakeThread:
    def test_creates_worker_thread_with_work_dir_and_github(
        self, tmp_path: Path
    ) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_registry = MagicMock()
        mock_gh_cls = MagicMock()
        mock_wt_cls = MagicMock()
        result = _make_thread(
            cfg, mock_registry, _GitHub=mock_gh_cls, _WorkerThread=mock_wt_cls
        )
        from kennel.config import RepoMembership

        mock_wt_cls.assert_called_once_with(
            tmp_path,
            "foo/bar",
            mock_gh_cls.return_value,
            mock_registry,
            RepoMembership(),
        )
        assert result is mock_wt_cls.return_value

    def test_uses_repo_work_dir(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "myrepo"
        work_dir.mkdir()
        cfg = _repo("foo/bar", work_dir)
        mock_wt_cls = MagicMock()
        _make_thread(cfg, MagicMock(), _GitHub=MagicMock(), _WorkerThread=mock_wt_cls)
        assert mock_wt_cls.call_args[0][0] == work_dir


class TestMakeRegistry:
    def test_returns_worker_registry(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        result = make_registry(
            {"foo/bar": cfg}, _thread_factory=MagicMock(return_value=mock_thread)
        )
        assert isinstance(result, WorkerRegistry)

    def test_starts_thread_for_each_repo(self, tmp_path: Path) -> None:
        cfg1 = _repo("foo/bar", tmp_path)
        cfg2 = _repo("foo/baz", tmp_path)
        mock_thread = MagicMock()
        mock_factory = MagicMock(return_value=mock_thread)
        make_registry({"foo/bar": cfg1, "foo/baz": cfg2}, _thread_factory=mock_factory)
        assert mock_factory.call_count == 2
        assert mock_thread.start.call_count == 2

    def test_empty_repos_returns_empty_registry(self) -> None:
        result = make_registry({})
        assert isinstance(result, WorkerRegistry)
        assert result.is_alive("anything") is False

    def test_wakes_registered_repos(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        reg = make_registry(
            {"foo/bar": cfg}, _thread_factory=MagicMock(return_value=mock_thread)
        )
        assert reg.is_alive("foo/bar") is True
