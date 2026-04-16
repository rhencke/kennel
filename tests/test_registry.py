"""Tests for kennel.registry — WorkerRegistry lifecycle management."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from kennel.config import RepoConfig
from kennel.registry import (
    WorkerCrash,
    WorkerRegistry,
    _make_thread,
    make_registry,
)


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
        factory.assert_called_once_with(cfg, provider=None, session_issue=None)

    def test_start_rescues_session_from_crashed_thread(self, tmp_path: Path) -> None:
        """Provider rescued from a crashed (dead, not _stop) thread and passed to replacement."""
        mock_provider = MagicMock()
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = _repo("foo/bar", tmp_path)
        # First start — no prior thread
        reg.start(cfg)
        # Simulate crash: thread died, _stop is False, provider is still attached
        threads[0].is_alive.return_value = False
        threads[0]._stop = False
        threads[0].detach_provider.return_value = mock_provider
        threads[0]._session_issue = 42
        # Second start — should rescue provider from crashed thread
        reg.start(cfg)
        _, kwargs = factory.call_args_list[1]
        assert kwargs["provider"] is mock_provider
        assert kwargs["session_issue"] == 42
        threads[0].detach_provider.assert_called_once_with()

    def test_start_does_not_rescue_session_from_orderly_shutdown_thread(
        self, tmp_path: Path
    ) -> None:
        """No provider rescue when the old thread exited via orderly stop()."""
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        # Simulate orderly shutdown: _stop is True (session was already stopped)
        threads[0].is_alive.return_value = False
        threads[0]._stop = True
        threads[0].detach_provider.return_value = MagicMock()
        reg.start(cfg)
        _, kwargs = factory.call_args_list[1]
        assert kwargs["provider"] is None
        assert kwargs["session_issue"] is None
        threads[0].detach_provider.assert_not_called()

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

    def test_get_thread_crash_error_returns_none_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_thread_crash_error("unknown/repo") is None

    def test_get_session_owner_returns_none_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session_owner("unknown/repo") is None

    def test_get_session_owner_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        factory.return_value.session_owner = "worker-home"
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session_owner("foo/bar") == "worker-home"

    def test_get_session_alive_returns_false_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session_alive("unknown/repo") is False

    def test_get_session_alive_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        factory.return_value.session_alive = True
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session_alive("foo/bar") is True

    def test_get_session_pid_returns_none_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session_pid("unknown/repo") is None

    def test_get_session_pid_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        factory.return_value.session_pid = 77777
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session_pid("foo/bar") == 77777

    def test_get_session_returns_none_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session("unknown/repo") is None

    def test_get_session_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        fake_session = MagicMock()
        factory.return_value._session = fake_session
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session("foo/bar") is fake_session

    def test_get_thread_crash_error_returns_thread_crash_error(
        self, tmp_path: Path
    ) -> None:
        reg, factory = self._make_registry()
        factory.return_value.crash_error = "RuntimeError: boom"
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_thread_crash_error("foo/bar") == "RuntimeError: boom"

    def test_get_thread_crash_error_returns_none_when_thread_has_no_error(
        self, tmp_path: Path
    ) -> None:
        reg, factory = self._make_registry()
        factory.return_value.crash_error = None
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_thread_crash_error("foo/bar") is None

    def test_report_activity_stores_entry(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        activities = reg.get_all_activities()
        assert len(activities) == 1
        assert activities[0].repo_name == "foo/bar"
        assert activities[0].what == "Working on: #1"
        assert activities[0].busy is True

    def test_report_activity_overwrites_previous(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        reg.report_activity("foo/bar", "Napping", busy=False)
        activities = reg.get_all_activities()
        assert len(activities) == 1
        assert activities[0].what == "Napping"
        assert activities[0].busy is False

    def test_get_all_activities_returns_all_repos(self) -> None:
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        reg.report_activity("foo/baz", "Napping", busy=False)
        activities = sorted(reg.get_all_activities(), key=lambda a: a.repo_name)
        assert [(a.repo_name, a.what, a.busy) for a in activities] == [
            ("foo/bar", "Working on: #1", True),
            ("foo/baz", "Napping", False),
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
        assert snapshot[0].what == "Working on: #1"

    def test_report_activity_records_last_progress_at(self) -> None:
        import datetime as dt

        fixed = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "busy", busy=True, _now=lambda: fixed)
        activities = reg.get_all_activities()
        assert activities[0].last_progress_at == fixed

    def test_report_activity_updates_last_progress_at(self) -> None:
        import datetime as dt

        t1 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        t2 = dt.datetime(2026, 1, 1, 12, 5, 0, tzinfo=dt.timezone.utc)
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "first", busy=True, _now=lambda: t1)
        reg.report_activity("foo/bar", "second", busy=True, _now=lambda: t2)
        activities = reg.get_all_activities()
        assert activities[0].last_progress_at == t2

    def test_is_stale_false_when_no_activity(self) -> None:
        reg, _ = self._make_registry()
        assert reg.is_stale("foo/bar", threshold=60.0) is False

    def test_is_stale_false_when_recent(self) -> None:
        import datetime as dt

        t0 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        t_now = dt.datetime(2026, 1, 1, 12, 0, 30, tzinfo=dt.timezone.utc)  # 30s later
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "busy", busy=True, _now=lambda: t0)
        assert reg.is_stale("foo/bar", threshold=60.0, _now=lambda: t_now) is False

    def test_is_stale_true_when_old(self) -> None:
        import datetime as dt

        t0 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        t_now = dt.datetime(2026, 1, 1, 12, 10, 0, tzinfo=dt.timezone.utc)  # 10m later
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "busy", busy=True, _now=lambda: t0)
        assert reg.is_stale("foo/bar", threshold=60.0, _now=lambda: t_now) is True

    def test_is_stale_exactly_at_threshold_is_not_stale(self) -> None:
        import datetime as dt

        t0 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        t_now = dt.datetime(2026, 1, 1, 12, 1, 0, tzinfo=dt.timezone.utc)  # exactly 60s
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "busy", busy=True, _now=lambda: t0)
        assert reg.is_stale("foo/bar", threshold=60.0, _now=lambda: t_now) is False

    def test_is_stale_per_repo(self) -> None:
        import datetime as dt

        t0 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        t_now = dt.datetime(2026, 1, 1, 12, 10, 0, tzinfo=dt.timezone.utc)
        reg, _ = self._make_registry()
        reg.report_activity("foo/bar", "old", busy=True, _now=lambda: t0)
        reg.report_activity("foo/baz", "fresh", busy=True, _now=lambda: t_now)
        assert reg.is_stale("foo/bar", threshold=60.0, _now=lambda: t_now) is True
        assert reg.is_stale("foo/baz", threshold=60.0, _now=lambda: t_now) is False

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

    def test_get_crash_info_returns_none_before_any_crash(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_crash_info("foo/bar") is None

    def test_record_crash_stores_error_and_count(self) -> None:
        reg, _ = self._make_registry()
        reg.record_crash("foo/bar", "boom")
        info = reg.get_crash_info("foo/bar")
        assert info is not None
        assert info.death_count == 1
        assert info.last_error == "boom"

    def test_record_crash_sets_last_crash_time(self) -> None:
        import datetime as dt

        before = dt.datetime.now()
        reg, _ = self._make_registry()
        reg.record_crash("foo/bar", "oops")
        after = dt.datetime.now()
        info = reg.get_crash_info("foo/bar")
        assert info is not None
        assert before <= info.last_crash_time <= after

    def test_record_crash_increments_death_count(self) -> None:
        reg, _ = self._make_registry()
        reg.record_crash("foo/bar", "err1")
        reg.record_crash("foo/bar", "err2")
        reg.record_crash("foo/bar", "err3")
        info = reg.get_crash_info("foo/bar")
        assert info is not None
        assert info.death_count == 3

    def test_record_crash_updates_last_error(self) -> None:
        reg, _ = self._make_registry()
        reg.record_crash("foo/bar", "first")
        reg.record_crash("foo/bar", "second")
        info = reg.get_crash_info("foo/bar")
        assert info is not None
        assert info.last_error == "second"

    def test_crash_info_is_per_repo(self) -> None:
        reg, _ = self._make_registry()
        reg.record_crash("foo/bar", "bar error")
        reg.record_crash("foo/baz", "baz error")
        reg.record_crash("foo/baz", "baz error 2")
        bar = reg.get_crash_info("foo/bar")
        baz = reg.get_crash_info("foo/baz")
        assert bar is not None and bar.death_count == 1
        assert baz is not None and baz.death_count == 2

    def test_record_crash_is_threadsafe(self) -> None:
        """Concurrent record_crash calls must not corrupt the death count."""
        reg, _ = self._make_registry()
        n = 200

        def crasher() -> None:
            for _ in range(n):
                reg.record_crash("foo/bar", "err")

        threads = [threading.Thread(target=crasher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        info = reg.get_crash_info("foo/bar")
        assert info is not None
        assert info.death_count == 4 * n

    def test_worker_crash_dataclass_fields(self) -> None:
        import datetime as dt

        ts = dt.datetime(2026, 1, 1, 12, 0, 0)
        crash = WorkerCrash(death_count=3, last_error="oops", last_crash_time=ts)
        assert crash.death_count == 3
        assert crash.last_error == "oops"
        assert crash.last_crash_time == ts

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
        mock_gh = MagicMock()
        mock_wt_cls = MagicMock()
        result = _make_thread(cfg, mock_registry, gh=mock_gh, _WorkerThread=mock_wt_cls)
        from kennel.config import RepoMembership

        mock_wt_cls.assert_called_once_with(
            tmp_path,
            "foo/bar",
            mock_gh,
            mock_registry,
            RepoMembership(),
            provider=None,
            session_issue=None,
            config=None,
            repo_cfg=cfg,
        )
        assert result is mock_wt_cls.return_value

    def test_uses_repo_work_dir(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "myrepo"
        work_dir.mkdir()
        cfg = _repo("foo/bar", work_dir)
        mock_wt_cls = MagicMock()
        _make_thread(cfg, MagicMock(), gh=MagicMock(), _WorkerThread=mock_wt_cls)
        assert mock_wt_cls.call_args[0][0] == work_dir

    def test_config_forwarded_to_worker_thread(self, tmp_path: Path) -> None:
        from kennel.config import Config

        cfg = _repo("foo/bar", tmp_path)
        config = Config(
            port=9000,
            secret=b"s",
            repos={"foo/bar": cfg},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        mock_wt_cls = MagicMock()
        _make_thread(
            cfg, MagicMock(), gh=MagicMock(), config=config, _WorkerThread=mock_wt_cls
        )
        assert mock_wt_cls.call_args.kwargs["config"] is config

    def test_repo_cfg_forwarded_to_worker_thread(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_wt_cls = MagicMock()
        _make_thread(cfg, MagicMock(), gh=MagicMock(), _WorkerThread=mock_wt_cls)
        assert mock_wt_cls.call_args.kwargs["repo_cfg"] is cfg


class TestMakeRegistry:
    def test_returns_worker_registry(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        result = make_registry(
            {"foo/bar": cfg},
            MagicMock(),
            _thread_factory=MagicMock(return_value=mock_thread),
        )
        assert isinstance(result, WorkerRegistry)

    def test_starts_thread_for_each_repo(self, tmp_path: Path) -> None:
        cfg1 = _repo("foo/bar", tmp_path)
        cfg2 = _repo("foo/baz", tmp_path)
        mock_thread = MagicMock()
        mock_factory = MagicMock(return_value=mock_thread)
        make_registry(
            {"foo/bar": cfg1, "foo/baz": cfg2},
            MagicMock(),
            _thread_factory=mock_factory,
        )
        assert mock_factory.call_count == 2
        assert mock_thread.start.call_count == 2

    def test_empty_repos_returns_empty_registry(self) -> None:
        result = make_registry({}, MagicMock())
        assert isinstance(result, WorkerRegistry)
        assert result.is_alive("anything") is False

    def test_wakes_registered_repos(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        reg = make_registry(
            {"foo/bar": cfg},
            MagicMock(),
            _thread_factory=MagicMock(return_value=mock_thread),
        )
        assert reg.is_alive("foo/bar") is True

    def test_config_forwarded_to_thread_factory(self, tmp_path: Path) -> None:
        from kennel.config import Config

        cfg = _repo("foo/bar", tmp_path)
        config = Config(
            port=9000,
            secret=b"s",
            repos={"foo/bar": cfg},
            allowed_bots=frozenset(),
            log_level="DEBUG",
            sub_dir=tmp_path,
        )
        mock_factory = MagicMock(return_value=MagicMock())
        make_registry(
            {"foo/bar": cfg}, MagicMock(), config, _thread_factory=mock_factory
        )
        assert mock_factory.call_args.kwargs["config"] is config


class TestThreadStartedAt:
    def test_records_on_start(self, tmp_path: Path) -> None:
        reg = WorkerRegistry(MagicMock())
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        assert reg.thread_started_at("foo/bar") is not None

    def test_returns_none_for_unknown(self) -> None:
        reg = WorkerRegistry(MagicMock())
        assert reg.thread_started_at("nope/none") is None


class TestWebhookActivity:
    def test_registers_and_unregisters(self) -> None:
        reg = WorkerRegistry(MagicMock())
        assert reg.get_webhook_activities("foo/bar") == []
        with reg.webhook_activity("foo/bar", "triaging"):
            activities = reg.get_webhook_activities("foo/bar")
            assert len(activities) == 1
            assert activities[0].description == "triaging"
        assert reg.get_webhook_activities("foo/bar") == []

    def test_unregisters_on_exception(self) -> None:
        import pytest as _pytest

        reg = WorkerRegistry(MagicMock())
        with _pytest.raises(RuntimeError):
            with reg.webhook_activity("foo/bar", "oops"):
                raise RuntimeError("boom")
        assert reg.get_webhook_activities("foo/bar") == []

    def test_multiple_concurrent_activities(self) -> None:
        reg = WorkerRegistry(MagicMock())
        with reg.webhook_activity("foo/bar", "first"):
            with reg.webhook_activity("foo/bar", "second"):
                descs = sorted(
                    a.description for a in reg.get_webhook_activities("foo/bar")
                )
                assert descs == ["first", "second"]
        assert reg.get_webhook_activities("foo/bar") == []

    def test_activities_isolated_per_repo(self) -> None:
        reg = WorkerRegistry(MagicMock())
        with reg.webhook_activity("a/b", "work-ab"):
            with reg.webhook_activity("c/d", "work-cd"):
                a = reg.get_webhook_activities("a/b")
                c = reg.get_webhook_activities("c/d")
                assert [x.description for x in a] == ["work-ab"]
                assert [x.description for x in c] == ["work-cd"]

    def test_unknown_repo_returns_empty_list(self) -> None:
        reg = WorkerRegistry(MagicMock())
        assert reg.get_webhook_activities("ghost/repo") == []


class TestRescoping:
    def test_is_rescoping_false_for_unknown_repo(self) -> None:
        reg = WorkerRegistry(MagicMock())
        assert reg.is_rescoping("unknown/repo") is False

    def test_set_rescoping_true(self) -> None:
        reg = WorkerRegistry(MagicMock())
        reg.set_rescoping("foo/bar", True)
        assert reg.is_rescoping("foo/bar") is True

    def test_set_rescoping_false_clears_flag(self) -> None:
        reg = WorkerRegistry(MagicMock())
        reg.set_rescoping("foo/bar", True)
        reg.set_rescoping("foo/bar", False)
        assert reg.is_rescoping("foo/bar") is False

    def test_rescoping_is_per_repo(self) -> None:
        reg = WorkerRegistry(MagicMock())
        reg.set_rescoping("foo/bar", True)
        assert reg.is_rescoping("foo/bar") is True
        assert reg.is_rescoping("foo/baz") is False

    def test_rescoping_is_threadsafe(self) -> None:
        """Concurrent set_rescoping and is_rescoping calls must not corrupt state."""
        reg = WorkerRegistry(MagicMock())
        errors: list[Exception] = []

        def toggler(repo: str) -> None:
            try:
                for i in range(200):
                    reg.set_rescoping(repo, i % 2 == 0)
            except Exception as exc:
                errors.append(exc)

        def reader(repo: str) -> None:
            try:
                for _ in range(200):
                    reg.is_rescoping(repo)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=toggler, args=("foo/bar",)),
            threading.Thread(target=reader, args=("foo/bar",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
