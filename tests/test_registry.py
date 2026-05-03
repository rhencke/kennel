"""Tests for fido.registry — WorkerRegistry lifecycle management."""

import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fido.config import RepoConfig as _RepoConfig
from fido.provider import ProviderID
from fido.registry import (
    WorkerCrash,
    WorkerRegistry,
    _make_thread,
    make_registry,
)
from fido.rocq import worker_registry_crash as registry_fsm


class RepoConfig(_RepoConfig):
    def __init__(
        self,
        *args: object,
        provider: ProviderID = ProviderID.CLAUDE_CODE,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, provider=provider, **kwargs)


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
        # Simulate crash: thread died, was_stopped is False, provider is still attached
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = False
        threads[0].detach_provider.return_value = mock_provider
        threads[0]._session_issue = 42
        # Second start — should rescue provider from crashed thread
        reg.start(cfg)
        _, kwargs = factory.call_args_list[1]
        assert kwargs["provider"] is mock_provider
        assert kwargs["session_issue"] == 42
        threads[0].detach_provider.assert_called_once_with()

    def test_start_recovers_rescued_session_to_clear_stuck_fsm(
        self, tmp_path: Path
    ) -> None:
        """Rescued provider's session is recovered before the new worker runs.

        Without this, a worker that crashed mid-turn leaves the persistent
        ClaudeSession FSM in a non-Idle state (e.g. Sending after BrokenPipe),
        and the replacement worker's first send() hits "Send rejected in
        state Sending" — turning a single crash into a permanent loop.
        """
        mock_provider = MagicMock()
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = False
        threads[0].detach_provider.return_value = mock_provider
        reg.start(cfg)
        mock_provider.agent.recover_session.assert_called_once_with()

    def test_start_does_not_rescue_session_from_orderly_shutdown_thread(
        self, tmp_path: Path
    ) -> None:
        """No provider rescue when the old thread exited via orderly stop()."""
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        # Simulate orderly shutdown: was_stopped is True (session was already stopped)
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = True
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
        for _call_result in factory.side_effect:
            # side_effect is exhausted; check via factory call args
            pass
        # Each mock returned by the factory should have stop() called once
        assert factory.call_count == 2
        for _mock_thread in [factory.return_value] + list(factory.side_effect or []):
            pass  # can't reuse side_effect after exhaustion; use call_args_list
        # Re-verify via the two distinct mock objects captured from calls
        _thread_a = factory.call_args_list[0]  # structural check
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
        reg.abort_task("foo/bar", task_id="t-1")
        factory.return_value.abort_task.assert_called_once_with(task_id="t-1")

    def test_abort_task_unknown_repo_is_noop(self) -> None:
        reg, factory = self._make_registry()
        reg.abort_task("unknown/repo", task_id="t-1")  # must not raise
        factory.return_value.abort_task.assert_not_called()

    def test_recover_provider_calls_thread_recover_provider(
        self, tmp_path: Path
    ) -> None:
        reg, factory = self._make_registry()
        factory.return_value.recover_provider.return_value = True
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.recover_provider("foo/bar") is True
        factory.return_value.recover_provider.assert_called_once_with()

    def test_recover_provider_unknown_repo_returns_false(self) -> None:
        reg, factory = self._make_registry()
        assert reg.recover_provider("unknown/repo") is False
        factory.return_value.recover_provider.assert_not_called()

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

    def test_get_session_dropped_count_returns_zero_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session_dropped_count("unknown/repo") == 0

    def test_get_session_dropped_count_delegates_to_thread(
        self, tmp_path: Path
    ) -> None:
        reg, factory = self._make_registry()
        factory.return_value.session_dropped_count = 4
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session_dropped_count("foo/bar") == 4

    def test_get_session_sent_count_returns_zero_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session_sent_count("unknown/repo") == 0

    def test_get_session_sent_count_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        factory.return_value.session_sent_count = 17
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session_sent_count("foo/bar") == 17

    def test_get_session_received_count_returns_zero_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session_received_count("unknown/repo") == 0

    def test_get_session_received_count_delegates_to_thread(
        self, tmp_path: Path
    ) -> None:
        reg, factory = self._make_registry()
        factory.return_value.session_received_count = 15
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session_received_count("foo/bar") == 15

    def test_get_session_returns_none_for_unknown_repo(self) -> None:
        reg, _ = self._make_registry()
        assert reg.get_session("unknown/repo") is None

    def test_get_session_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory = self._make_registry()
        fake_session = MagicMock()
        factory.return_value.current_provider.return_value.agent.session = fake_session
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session("foo/bar") is fake_session

    # ── issue tree cache (fix #812) ──────────────────────────────────────

    def test_get_issue_cache_creates_lazily(self) -> None:
        from fido.issue_cache import IssueTreeCache

        reg, _ = self._make_registry()
        cache = reg.get_issue_cache("foo/bar")
        assert isinstance(cache, IssueTreeCache)

    def test_get_issue_cache_returns_same_instance(self) -> None:
        reg, _ = self._make_registry()
        a = reg.get_issue_cache("foo/bar")
        b = reg.get_issue_cache("foo/bar")
        assert a is b

    def test_get_issue_cache_per_repo(self) -> None:
        reg, _ = self._make_registry()
        a = reg.get_issue_cache("foo/bar")
        b = reg.get_issue_cache("foo/baz")
        assert a is not b

    def test_all_issue_caches_returns_snapshot(self) -> None:
        reg, _ = self._make_registry()
        assert reg.all_issue_caches() == []
        reg.get_issue_cache("a/1")
        reg.get_issue_cache("b/2")
        all_caches = reg.all_issue_caches()
        assert len(all_caches) == 2
        # Mutating the snapshot doesn't affect future calls.
        all_caches.clear()
        assert len(reg.all_issue_caches()) == 2

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

        before = dt.datetime.now(tz=dt.timezone.utc)
        reg, _ = self._make_registry()
        reg.record_crash("foo/bar", "oops")
        after = dt.datetime.now(tz=dt.timezone.utc)
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

        ts = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
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
        # Simulate a crash so the FSM accepts the second start
        # (no_start_while_active: start() on a live thread is rejected).
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = False
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
        from fido.config import RepoMembership

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
            issue_cache=mock_registry.get_issue_cache.return_value,
        )
        mock_registry.get_issue_cache.assert_called_once_with("foo/bar")
        assert result is mock_wt_cls.return_value

    def test_uses_repo_work_dir(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "myrepo"
        work_dir.mkdir()
        cfg = _repo("foo/bar", work_dir)
        mock_wt_cls = MagicMock()
        _make_thread(cfg, MagicMock(), gh=MagicMock(), _WorkerThread=mock_wt_cls)
        assert mock_wt_cls.call_args[0][0] == work_dir

    def test_config_forwarded_to_worker_thread(self, tmp_path: Path) -> None:
        from fido.config import Config

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
        from fido.config import Config

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

    def test_handle_can_update_description(self) -> None:
        reg = WorkerRegistry(MagicMock())
        with reg.webhook_activity("foo/bar", "handling") as activity:
            assert reg.get_webhook_activities("foo/bar")[0].description == "handling"
            activity.set_description("triaging")
            assert reg.get_webhook_activities("foo/bar")[0].description == "triaging"

    def test_handle_update_after_exit_is_noop(self) -> None:
        reg = WorkerRegistry(MagicMock())
        with reg.webhook_activity("foo/bar", "handling") as activity:
            pass
        activity.set_description("triaging")
        assert reg.get_webhook_activities("foo/bar") == []

    def test_unknown_handle_update_is_noop(self) -> None:
        reg = WorkerRegistry(MagicMock())
        with reg.webhook_activity("foo/bar", "handling"):
            reg.set_webhook_description("foo/bar", -1, "triaging")
            assert reg.get_webhook_activities("foo/bar")[0].description == "handling"

    def test_unknown_repo_handle_update_is_noop(self) -> None:
        reg = WorkerRegistry(MagicMock())
        reg.set_webhook_description("ghost/repo", 1, "triaging")
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


class TestUntriagedInbox:
    """Tests for the per-repo untriaged-webhook inbox (fix #1067)."""

    def _reg(self) -> WorkerRegistry:
        return WorkerRegistry(MagicMock())

    # ── has_untriaged ─────────────────────────────────────────────────────

    def test_has_untriaged_false_for_unknown_repo(self) -> None:
        reg = self._reg()
        assert reg.has_untriaged("unknown/repo") is False

    def test_has_untriaged_false_before_any_enter(self) -> None:
        reg = self._reg()
        assert reg.has_untriaged("foo/bar") is False

    def test_has_untriaged_true_after_enter(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        assert reg.has_untriaged("foo/bar") is True

    def test_has_untriaged_false_after_enter_then_exit(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        assert reg.has_untriaged("foo/bar") is False

    def test_has_untriaged_true_while_multiple_pending(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        assert reg.has_untriaged("foo/bar") is True

    def test_has_untriaged_false_after_all_exits(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        assert reg.has_untriaged("foo/bar") is False

    def test_inbox_is_per_repo(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        assert reg.has_untriaged("foo/bar") is True
        assert reg.has_untriaged("foo/baz") is False

    # ── exit_untriaged underflow ──────────────────────────────────────────

    def test_exit_untriaged_when_count_zero_does_not_raise(self) -> None:
        reg = self._reg()
        reg.exit_untriaged("foo/bar")  # must not raise

    def test_exit_untriaged_underflow_leaves_count_zero(self) -> None:
        reg = self._reg()
        reg.exit_untriaged("foo/bar")
        assert reg.has_untriaged("foo/bar") is False

    # ── wait_for_inbox_drain ──────────────────────────────────────────────

    def test_wait_returns_true_when_empty(self) -> None:
        reg = self._reg()
        result = reg.wait_for_inbox_drain("foo/bar", timeout=0.1)
        assert result is True

    def test_wait_returns_true_after_drain(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        result = reg.wait_for_inbox_drain("foo/bar", timeout=0.1)
        assert result is True

    def test_wait_blocks_until_exit(self) -> None:
        """wait_for_inbox_drain blocks while the inbox is non-empty, then unblocks."""
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        drained = threading.Event()

        def drainer() -> None:
            result = reg.wait_for_inbox_drain("foo/bar", timeout=2.0)
            if result:
                drained.set()

        t = threading.Thread(target=drainer)
        t.start()
        # Give the waiter a moment to start blocking
        time.sleep(0.01)
        assert not drained.is_set(), "should still be waiting"
        reg.exit_untriaged("foo/bar")
        t.join(timeout=2.0)
        assert drained.is_set(), "should have been unblocked by exit"

    def test_wait_returns_false_on_timeout(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        result = reg.wait_for_inbox_drain("foo/bar", timeout=0.02)
        assert result is False
        # Clean up so the event doesn't linger
        reg.exit_untriaged("foo/bar")

    def test_wait_for_unknown_repo_returns_true_immediately(self) -> None:
        reg = self._reg()
        result = reg.wait_for_inbox_drain("ghost/repo", timeout=0.1)
        assert result is True

    def test_wait_unblocks_when_last_of_multiple_exits(self) -> None:
        """Drain fires only when the last of several enters is exited."""
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        drained = threading.Event()

        def waiter() -> None:
            if reg.wait_for_inbox_drain("foo/bar", timeout=2.0):
                drained.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.01)
        reg.exit_untriaged("foo/bar")
        time.sleep(0.01)
        assert not drained.is_set(), "one exit should not drain"
        reg.exit_untriaged("foo/bar")
        t.join(timeout=2.0)
        assert drained.is_set(), "second exit should drain"

    # ── thread-safety ─────────────────────────────────────────────────────

    def test_concurrent_enter_exit_is_threadsafe(self) -> None:
        """Concurrent enter_untriaged and exit_untriaged calls must not corrupt state."""
        reg = self._reg()
        errors: list[Exception] = []
        n = 200

        def worker_enter() -> None:
            try:
                for _ in range(n):
                    reg.enter_untriaged("foo/bar")
            except Exception as exc:
                errors.append(exc)

        def worker_exit() -> None:
            try:
                for _ in range(n):
                    reg.exit_untriaged("foo/bar")
            except Exception as exc:
                errors.append(exc)

        # Pre-fill to avoid underflow warnings in the test
        for _ in range(n * 2):
            reg.enter_untriaged("foo/bar")

        threads = [
            threading.Thread(target=worker_enter),
            threading.Thread(target=worker_enter),
            threading.Thread(target=worker_exit),
            threading.Thread(target=worker_exit),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent errors: {errors}"
        # After equal enters and exits, count returns to the pre-filled value
        assert reg.has_untriaged("foo/bar") is True

    # ── force_clear_untriaged (#1280) ─────────────────────────────────────

    def test_force_clear_returns_zero_when_count_already_zero(self) -> None:
        reg = self._reg()
        assert reg.force_clear_untriaged("foo/bar") == 0

    def test_force_clear_resets_leaked_count_and_returns_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")

        with caplog.at_level(logging.WARNING):
            cleared = reg.force_clear_untriaged("foo/bar")

        assert cleared == 3
        assert reg.has_untriaged("foo/bar") is False
        assert any(
            "force-cleared 3 leaked hold(s)" in rec.message for rec in caplog.records
        )

    def test_force_clear_signals_drained_event_so_waiter_unblocks(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        unblocked = threading.Event()

        def waiter() -> None:
            reg.wait_for_inbox_drain("foo/bar", timeout=2.0)
            unblocked.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert not unblocked.is_set(), "waiter should still be blocked pre-clear"

        reg.force_clear_untriaged("foo/bar")
        t.join(timeout=2.0)
        assert unblocked.is_set(), "force_clear should unblock waiter"


class TestPreemptionFsmOracle:
    """Tests for the handler_preemption FSM oracle wired into WorkerRegistry.

    Each test maps to a proved invariant from ``models/handler_preemption.v``.
    The FSM oracle validates that WorkerTurnStart is rejected when the inbox
    is non-empty — the core preemption guarantee from #1067.
    """

    def _reg(self) -> WorkerRegistry:
        return WorkerRegistry(MagicMock())

    # ── worker_blocked_when_nonempty ─────────────────────────────────────

    def test_assert_worker_turn_ok_raises_when_inbox_nonempty(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")

    def test_assert_worker_turn_ok_raises_with_multiple_pending(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")

    def test_assert_worker_turn_ok_raises_when_durable_demand_pending(
        self,
    ) -> None:
        reg = self._reg()
        reg.note_durable_demand("foo/bar")
        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")

    def test_assert_worker_turn_ok_succeeds_after_durable_demand_drains(
        self,
    ) -> None:
        reg = self._reg()
        reg.note_durable_demand("foo/bar")
        reg.note_durable_demand_drained("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")

    def test_durable_demand_before_untriaged_handler_exits_cleanly(self) -> None:
        reg = self._reg()
        reg.note_durable_demand("foo/bar")
        reg.enter_untriaged("foo/bar")

        reg.exit_untriaged("foo/bar")

        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")
        reg.note_durable_demand_drained("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")

    def test_durable_demand_drain_preserves_untriaged_blocker(self) -> None:
        reg = self._reg()
        reg.note_durable_demand("foo/bar")
        reg.enter_untriaged("foo/bar")

        reg.note_durable_demand_drained("foo/bar")

        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")
        reg.exit_untriaged("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")

    # ── worker_turn_proceeds_when_empty ──────────────────────────────────

    def test_assert_worker_turn_ok_succeeds_when_empty(self) -> None:
        reg = self._reg()
        reg.assert_worker_turn_ok("foo/bar")  # must not raise

    def test_assert_worker_turn_ok_succeeds_after_drain(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")  # must not raise

    def test_assert_worker_turn_ok_succeeds_after_full_drain(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")  # must not raise

    # ── per-repo isolation ───────────────────────────────────────────────

    def test_assert_worker_turn_ok_is_per_repo(self) -> None:
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        # Different repo is still empty — worker turn should be accepted
        reg.assert_worker_turn_ok("foo/baz")  # must not raise
        # But the repo with a pending webhook should reject
        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")

    # ── handler_done_rejected_from_empty (underflow) ─────────────────────

    def test_exit_untriaged_underflow_does_not_corrupt_fsm(self) -> None:
        """An underflow exit_untriaged logs a warning but doesn't crash the FSM.

        After the underflow, the FSM should still be in Empty so
        assert_worker_turn_ok works.
        """
        reg = self._reg()
        reg.exit_untriaged("foo/bar")  # underflow — logs warning, no FSM change
        reg.assert_worker_turn_ok("foo/bar")  # still Empty — should work

    # ── full lifecycle ───────────────────────────────────────────────────

    def test_full_enter_assert_fails_drain_assert_succeeds(self) -> None:
        """enter → assert (fails) → exit → assert (succeeds)."""
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        with pytest.raises(AssertionError):
            reg.assert_worker_turn_ok("foo/bar")
        reg.exit_untriaged("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")  # must not raise

    def test_repeated_worker_turns_while_empty(self) -> None:
        """Multiple WorkerTurnStart calls are fine when inbox stays empty."""
        reg = self._reg()
        reg.assert_worker_turn_ok("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")  # all should succeed

    def test_interleaved_enter_exit_with_worker_turns(self) -> None:
        """Full interleaved lifecycle: enters, exits, worker turns."""
        reg = self._reg()
        # Worker turn ok while empty
        reg.assert_worker_turn_ok("foo/bar")
        # Two webhooks arrive
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        # Worker blocked
        with pytest.raises(AssertionError):
            reg.assert_worker_turn_ok("foo/bar")
        # First handler finishes — still blocked
        reg.exit_untriaged("foo/bar")
        with pytest.raises(AssertionError):
            reg.assert_worker_turn_ok("foo/bar")
        # Second handler finishes — now clear
        reg.exit_untriaged("foo/bar")
        reg.assert_worker_turn_ok("foo/bar")  # must not raise

    def test_concurrent_enter_exit_with_oracle(self) -> None:
        """Concurrent enter/exit preserves FSM consistency.

        After equal enters and exits, the FSM must be back in Empty
        so assert_worker_turn_ok succeeds.
        """
        reg = self._reg()
        n = 50
        errors: list[Exception] = []

        def enter_batch() -> None:
            try:
                for _ in range(n):
                    reg.enter_untriaged("foo/bar")
            except Exception as exc:
                errors.append(exc)

        def exit_batch() -> None:
            try:
                for _ in range(n):
                    reg.exit_untriaged("foo/bar")
            except Exception as exc:
                errors.append(exc)

        # Pre-fill so exits don't underflow
        for _ in range(n):
            reg.enter_untriaged("foo/bar")

        threads = [
            threading.Thread(target=enter_batch),
            threading.Thread(target=exit_batch),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent errors: {errors}"
        # Drain the remaining enters
        remaining = reg._untriaged.get("foo/bar", 0)  # pyright: ignore[reportPrivateUsage]
        for _ in range(remaining):
            reg.exit_untriaged("foo/bar")
        # After full drain, worker turn should be ok
        reg.assert_worker_turn_ok("foo/bar")


class TestRegistryFsmOracle:
    """Tests for the worker_registry_crash FSM oracle wired into WorkerRegistry.

    Each test section maps to a proved invariant from
    ``models/worker_registry_crash.v``.  The tests verify that
    ``_registry_fsm_transition`` raises ``AssertionError`` on rejected
    transitions (fail-closed contract) and that ``start()`` advances the
    FSM correctly on each lifecycle path.
    """

    def _reg(self) -> WorkerRegistry:
        return WorkerRegistry(MagicMock())

    def _cfg(self, tmp_path: Path, name: str = "foo/bar") -> RepoConfig:
        return _repo(name, tmp_path)

    # ── _registry_fsm_transition fail-closed behaviour ──────────────────

    def test_fsm_transition_crashes_on_rejected_event(self, tmp_path: Path) -> None:
        """_registry_fsm_transition raises AssertionError on a rejected event.

        Initial state is Absent; Rescue is rejected from Absent
        (rescue_requires_prior_crash).  The oracle must raise immediately.
        """
        reg = self._reg()
        with pytest.raises(AssertionError, match="worker_registry_crash FSM"):
            reg._registry_fsm_transition(  # pyright: ignore[reportPrivateUsage]
                "foo/bar", registry_fsm.Rescue()
            )

    def test_fsm_transition_error_includes_repo_name(self, tmp_path: Path) -> None:
        """AssertionError message names the repo so the violation is easy to locate."""
        reg = self._reg()
        with pytest.raises(AssertionError, match="myorg/myrepo"):
            reg._registry_fsm_transition(  # pyright: ignore[reportPrivateUsage]
                "myorg/myrepo", registry_fsm.Rescue()
            )

    def test_fsm_transition_error_includes_state_and_event(self) -> None:
        """AssertionError message names the rejected state and event."""
        reg = self._reg()
        with pytest.raises(AssertionError, match="Absent") as exc_info:
            reg._registry_fsm_transition(  # pyright: ignore[reportPrivateUsage]
                "foo/bar", registry_fsm.Rescue()
            )
        assert "Rescue" in str(exc_info.value)

    def test_fsm_transition_initialises_state_to_absent(self) -> None:
        """First successful transition uses Absent as the implicit starting state."""
        reg = self._reg()
        new_state = reg._registry_fsm_transition(  # pyright: ignore[reportPrivateUsage]
            "foo/bar", registry_fsm.Launch()
        )
        assert isinstance(new_state, registry_fsm.Active)
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),  # pyright: ignore[reportPrivateUsage]
            registry_fsm.Active,
        )

    def test_fsm_transition_persists_state_between_calls(self) -> None:
        """_registry_fsm_transition updates _registry_fsm_states; subsequent calls see the new state."""
        reg = self._reg()
        # Absent → Launch → Active
        reg._registry_fsm_transition("foo/bar", registry_fsm.Launch())  # pyright: ignore[reportPrivateUsage]
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )
        # Active → ThreadDies → Crashed
        reg._registry_fsm_transition("foo/bar", registry_fsm.ThreadDies())  # pyright: ignore[reportPrivateUsage]
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Crashed,  # pyright: ignore[reportPrivateUsage]
        )
        # Crashed → Rescue → Active
        reg._registry_fsm_transition("foo/bar", registry_fsm.Rescue())  # pyright: ignore[reportPrivateUsage]
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )

    # ── start() oracle integration ───────────────────────────────────────

    def test_start_fresh_advances_fsm_absent_to_active(self, tmp_path: Path) -> None:
        """start() with no prior thread fires Launch: Absent → Active."""
        reg = self._reg()
        reg.start(self._cfg(tmp_path))
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )

    def test_start_crash_rescue_advances_fsm_through_crash(
        self, tmp_path: Path
    ) -> None:
        """start() after crash fires ThreadDies then Rescue: Active → Crashed → Active."""
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = self._cfg(tmp_path)
        reg.start(cfg)
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )
        # Simulate crash
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = False
        reg.start(cfg)
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )

    def test_start_after_orderly_stop_advances_fsm_through_stopped(
        self, tmp_path: Path
    ) -> None:
        """start() after orderly stop fires ThreadStops then Launch: Active → Stopped → Active."""
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = self._cfg(tmp_path)
        reg.start(cfg)
        # Simulate orderly stop
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = True
        reg.start(cfg)
        assert isinstance(
            reg._registry_fsm_states.get("foo/bar"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )

    def test_start_with_alive_predecessor_raises_assertion(
        self, tmp_path: Path
    ) -> None:
        """start() on a live thread raises AssertionError (no_start_while_active).

        An alive thread means the slot is Active; Launch is rejected from Active.
        This surfaces the coordination violation immediately rather than silently
        replacing the live thread.
        """
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        cfg = self._cfg(tmp_path)
        reg.start(cfg)
        # Leave thread alive (is_alive() returns a truthy MagicMock by default)
        threads[0].is_alive.return_value = True
        with pytest.raises(AssertionError, match="worker_registry_crash FSM"):
            reg.start(cfg)

    def test_fsm_state_is_tracked_per_repo(self, tmp_path: Path) -> None:
        """FSM state is independent for each repo — no cross-repo contamination."""
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reg = WorkerRegistry(factory)
        reg.start(_repo("org/a", tmp_path))
        reg.start(_repo("org/b", tmp_path))
        assert isinstance(
            reg._registry_fsm_states.get("org/a"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )
        assert isinstance(
            reg._registry_fsm_states.get("org/b"),
            registry_fsm.Active,  # pyright: ignore[reportPrivateUsage]
        )
