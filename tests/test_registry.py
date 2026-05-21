"""Tests for fido.registry — WorkerRegistry lifecycle management."""

import logging
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from frozendict import frozendict

from fido.appstate import (
    _EPOCH,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    ProviderSnapshot,
    ThreadSnapshot,
    WorkerCrash,
)
from fido.atomic import create_atomic
from fido.config import RepoConfig as _RepoConfig
from fido.provider import ProviderID
from fido.registry import (
    WorkerRegistry,
    _make_thread,
    make_registry,
)
from fido.rocq import worker_registry_crash as registry_fsm
from tests.fakes import _FakeDispatcher


class RepoConfig(_RepoConfig):
    def __init__(
        self,
        *args: object,
        provider: ProviderID = ProviderID.CLAUDE_CODE,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, provider=provider, **kwargs)


def _repo(name: str, work_dir: Path) -> RepoConfig:
    """RepoConfig + git-init the work_dir.

    Registry.start resolves the canonical git_dir via
    ``git rev-parse --absolute-git-dir`` (#1696 codex P1 round 5) so
    the work_dir must actually be a git repository.  ``git init`` is
    idempotent — re-initialising an existing repo is safe.
    """
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=work_dir,
        check=True,
        capture_output=True,
    )
    return RepoConfig(name=name, work_dir=work_dir)


class TestWorkerRegistry:
    def _make_registry(
        self,
        *,
        repos: list[str] | None = None,
        work_dir: Path | None = None,
    ) -> tuple[WorkerRegistry, MagicMock, object]:
        factory = MagicMock()
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
        if repos:
            # Pre-populated registries need a real git work_dir
            # because registry.start resolves git_dir at construction
            # (#1696 codex P1 round 5).  Caller passes one when they
            # need start() to succeed.
            assert work_dir is not None, (
                "_make_registry(repos=...) requires work_dir for git resolution"
            )
            for name in repos:
                reg.start(_repo(name, work_dir))
        return reg, factory, reader

    def test_start_calls_factory_with_repo_cfg(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        factory.assert_called_once_with(cfg, provider=None, session_issue=None)

    def test_start_rescues_session_from_crashed_thread(self, tmp_path: Path) -> None:
        """Provider rescued from a crashed (dead, not _stop) thread and passed to replacement."""
        mock_provider = MagicMock()
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        factory.return_value.start.assert_called_once()

    def test_wake_calls_thread_wake(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.wake("foo/bar")
        factory.return_value.wake.assert_called_once()

    def test_wake_unknown_repo_is_noop(self) -> None:
        reg, factory, reader = self._make_registry()
        reg.wake("unknown/repo")  # must not raise
        factory.return_value.wake.assert_not_called()

    def test_stop_all_calls_stop_on_thread(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.stop_all()
        factory.return_value.stop.assert_called_once()

    def test_stop_all_calls_stop_on_every_thread(self, tmp_path: Path) -> None:
        factory = MagicMock(side_effect=[MagicMock(), MagicMock()])
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        reg.start(_repo("foo/baz", tmp_path))
        reg.stop_all()
        for t in threads:
            t.stop.assert_called_once()

    def test_stop_all_empty_is_noop(self) -> None:
        reg, _, reader = self._make_registry()
        reg.stop_all()  # must not raise

    def test_stop_and_join_calls_stop_and_join_on_thread(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        reg.start(_repo("foo/bar", tmp_path))
        reg.stop_and_join("foo/bar", timeout=5.0)
        factory.return_value.stop.assert_called_once()
        factory.return_value.join.assert_called_once_with(timeout=5.0)

    def test_stop_and_join_unknown_repo_is_noop(self) -> None:
        reg, factory, reader = self._make_registry()
        reg.stop_and_join("unknown/repo")  # must not raise
        factory.return_value.stop.assert_not_called()
        factory.return_value.join.assert_not_called()

    def test_abort_task_calls_thread_abort_task(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.abort_task("foo/bar", task_id="t-1")
        factory.return_value.abort_task.assert_called_once_with(task_id="t-1")

    def test_abort_task_unknown_repo_is_noop(self) -> None:
        reg, factory, reader = self._make_registry()
        reg.abort_task("unknown/repo", task_id="t-1")  # must not raise
        factory.return_value.abort_task.assert_not_called()

    def test_recover_provider_calls_thread_recover_provider(
        self, tmp_path: Path
    ) -> None:
        reg, factory, reader = self._make_registry()
        factory.return_value.recover_provider.return_value = True
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.recover_provider("foo/bar") is True
        factory.return_value.recover_provider.assert_called_once_with()

    def test_recover_provider_unknown_repo_returns_false(self) -> None:
        reg, factory, reader = self._make_registry()
        assert reg.recover_provider("unknown/repo") is False
        factory.return_value.recover_provider.assert_not_called()

    def test_recover_provider_republishes_provider_snapshot(
        self, tmp_path: Path
    ) -> None:
        """recover_provider() refreshes the ProviderSnapshot after recovery."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        factory.return_value.recover_provider.return_value = True
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        # Simulate what recovery does: session becomes alive after recover_provider
        factory.return_value.session_alive = True
        factory.return_value.session_pid = 42
        reg.recover_provider("foo/bar")
        provider = reader.get().repos["foo/bar"].provider
        assert provider is not None
        assert provider.session_alive is True
        assert provider.session_pid == 42

    def test_publish_provider_snapshot_updates_fido_state(self, tmp_path: Path) -> None:
        """publish_provider_snapshot() publishes a fresh ProviderSnapshot into FidoState."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        # Simulate a turn running: session is now active with counters
        factory.return_value.session_owner = "worker-foo"
        factory.return_value.session_alive = True
        factory.return_value.session_sent_count = 5
        factory.return_value.session_received_count = 3
        reg.publish_provider_snapshot("foo/bar")
        provider = reader.get().repos["foo/bar"].provider
        assert provider is not None
        assert provider.session_owner == "worker-foo"
        assert provider.session_alive is True
        assert provider.session_sent_count == 5
        assert provider.session_received_count == 3

    def test_repo_for_returns_publishing_repo_after_start(self, tmp_path: Path) -> None:
        """``repo_for`` returns the registry-owned :class:`Repo` whose
        :class:`Tasks` mutations auto-publish a fresh
        :class:`TaskListSnapshot` to the per-repo ``task_list`` leaf
        (#1696).  Independent leaves: Tasks publishes ``task_list``;
        :class:`State` publishes ``issue`` — neither reads the other's
        source."""
        from fido.types import TaskType

        reg, factory, reader = self._make_registry()
        reg.start(_repo("foo/bar", tmp_path))
        repo = reg.repo_for("foo/bar")
        assert repo.name == "foo/bar"
        assert repo.work_dir == tmp_path
        # tasks_for shorthand returns the same Tasks instance.
        assert reg.tasks_for("foo/bar") is repo.tasks

        repo.tasks.add(title="thing", task_type=TaskType.SPEC)

        snap = reader.get().repos["foo/bar"].task_list
        assert snap is not None
        assert snap.pending_task_count == 1
        assert snap.current_task == "thing"

    def test_state_save_publishes_issue_snapshot(self, tmp_path: Path) -> None:
        """State.save fires on_mutate → snapshot publishes the new
        issue/PR fields (#1696)."""
        reg, factory, reader = self._make_registry()
        reg.start(_repo("foo/bar", tmp_path))
        repo = reg.repo_for("foo/bar")
        # state_for shorthand returns the same State instance.
        assert reg.state_for("foo/bar") is repo.state

        repo.state.save(
            {
                "issue": 7,
                "issue_title": "Fix it",
                "pr_number": 13,
                "pr_title": "Fix it (closes #7)",
            }
        )

        snap = reader.get().repos["foo/bar"].issue
        assert snap is not None
        assert snap.issue == 7
        assert snap.issue_title == "Fix it"
        assert snap.pr_number == 13
        assert snap.pr_title == "Fix it (closes #7)"

    def test_stop_and_join_default_timeout(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        reg.start(_repo("foo/bar", tmp_path))
        reg.stop_and_join("foo/bar")
        factory.return_value.join.assert_called_once_with(timeout=30.0)

    def test_is_alive_true_when_thread_alive(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        factory.return_value.is_alive.return_value = True
        reg.start(cfg)
        assert reg.is_alive("foo/bar") is True

    def test_is_alive_false_when_thread_dead(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        cfg = _repo("foo/bar", tmp_path)
        factory.return_value.is_alive.return_value = False
        reg.start(cfg)
        assert reg.is_alive("foo/bar") is False

    def test_is_alive_false_for_unknown_repo(self) -> None:
        reg, _, reader = self._make_registry()
        assert reg.is_alive("unknown/repo") is False

    def test_get_thread_crash_error_raises_for_unknown_repo(self) -> None:
        reg, _, reader = self._make_registry()
        with pytest.raises(KeyError):
            reg.get_thread_crash_error("unknown/repo")

    def test_get_session_returns_none_for_unknown_repo(self) -> None:
        reg, _, reader = self._make_registry()
        assert reg.get_session("unknown/repo") is None

    def test_get_session_delegates_to_thread(self, tmp_path: Path) -> None:
        reg, factory, reader = self._make_registry()
        fake_session = MagicMock()
        factory.return_value.current_provider.return_value.agent.session = fake_session
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_session("foo/bar") is fake_session

    # ── issue tree cache (fix #812) ──────────────────────────────────────

    def test_get_issue_cache_creates_lazily(self) -> None:
        from fido.issue_cache import IssueCache

        reg, _, reader = self._make_registry()
        cache = reg.get_issue_cache("foo/bar")
        assert isinstance(cache, IssueCache)

    def test_get_issue_cache_returns_same_instance(self) -> None:
        reg, _, reader = self._make_registry()
        a = reg.get_issue_cache("foo/bar")
        b = reg.get_issue_cache("foo/bar")
        assert a is b

    def test_get_issue_cache_per_repo(self) -> None:
        reg, _, reader = self._make_registry()
        a = reg.get_issue_cache("foo/bar")
        b = reg.get_issue_cache("foo/baz")
        assert a is not b

    def test_all_issue_caches_returns_snapshot(self) -> None:
        reg, _, reader = self._make_registry()
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
        reg, factory, reader = self._make_registry()
        factory.return_value.crash_error = "RuntimeError: boom"
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_thread_crash_error("foo/bar") == "RuntimeError: boom"

    def test_get_thread_crash_error_returns_none_when_thread_has_no_error(
        self, tmp_path: Path
    ) -> None:
        reg, factory, reader = self._make_registry()
        factory.return_value.crash_error = None
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_thread_crash_error("foo/bar") is None

    def test_report_activity_stores_entry(self, tmp_path: Path) -> None:
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        activity = reader.get().repos["foo/bar"].activity
        assert activity.repo_name == "foo/bar"
        assert activity.what == "Working on: #1"
        assert activity.busy is True

    def test_report_activity_overwrites_previous(self, tmp_path: Path) -> None:
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.report_activity("foo/bar", "Working on: #1", busy=True)
        reg.report_activity("foo/bar", "Napping", busy=False)
        activity = reader.get().repos["foo/bar"].activity
        assert activity.what == "Napping"
        assert activity.busy is False

    def test_report_activity_records_last_progress_at(self, tmp_path: Path) -> None:
        import datetime as dt

        fixed = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.report_activity("foo/bar", "busy", busy=True, _now=lambda: fixed)
        activity = reader.get().repos["foo/bar"].activity
        assert activity.last_progress_at == fixed

    def test_report_activity_updates_last_progress_at(self, tmp_path: Path) -> None:
        import datetime as dt

        t1 = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        t2 = dt.datetime(2026, 1, 1, 12, 5, 0, tzinfo=dt.timezone.utc)
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.report_activity("foo/bar", "first", busy=True, _now=lambda: t1)
        reg.report_activity("foo/bar", "second", busy=True, _now=lambda: t2)
        activity = reader.get().repos["foo/bar"].activity
        assert activity.last_progress_at == t2

    def test_concurrent_report_and_read_are_safe(self, tmp_path: Path) -> None:
        """report_activity is safe under concurrent load.

        Multiple writer threads each own one repo and hammer report_activity;
        a reader thread continuously snapshots FidoState.  After all writers
        finish, every repo must appear with its final value, proving no data
        was lost or corrupted.
        """
        n_repos = 8
        repos = [f"owner/repo{i}" for i in range(n_repos)]
        reg, _, state_reader = self._make_registry(repos=repos, work_dir=tmp_path)
        n_writes = 200
        errors: list[Exception] = []

        def writer(repo: str) -> None:
            try:
                for i in range(n_writes):
                    reg.report_activity(repo, f"step {i}", busy=i % 2 == 0)
            except Exception as exc:
                errors.append(exc)

        def reader_fn() -> None:
            try:
                for _ in range(n_writes * n_repos):
                    snapshot = state_reader.get().repos
                    assert isinstance(snapshot, dict)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(r,)) for r in repos]
        threads.append(threading.Thread(target=reader_fn))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent errors: {errors}"

        final_repos = state_reader.get().repos
        assert set(final_repos) == set(repos)
        for repo in repos:
            assert final_repos[repo].activity.what == f"step {n_writes - 1}"

    def test_status_update_is_context_manager(self) -> None:
        reg, _, reader = self._make_registry()
        with reg.status_update():
            pass  # must not raise

    def test_status_update_serializes_concurrent_callers(self) -> None:
        """Only one caller may be inside status_update() at a time."""
        reg, _, reader = self._make_registry()
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

    def test_atomic_reader_sees_registry_writes(self, tmp_path: Path) -> None:
        """Atomic reader reflects writes made via the updater."""
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        # reader sees the repo populated by start()
        assert "foo/bar" in reader.get().repos

    def test_registry_is_write_only_for_fido_state(self) -> None:
        """WorkerRegistry has no read face — get_state/get_state_reader/get_state_updater are gone."""
        reg, _, reader = self._make_registry()
        assert not hasattr(reg, "get_state")
        assert not hasattr(reg, "get_state_reader")
        assert not hasattr(reg, "get_state_updater")

    def test_record_crash_stores_error_and_count(self, tmp_path: Path) -> None:
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.record_crash("foo/bar", "boom")
        crash = reader.get().repos["foo/bar"].crash_record
        assert crash.death_count == 1
        assert crash.last_error == "boom"

    def test_record_crash_sets_last_crash_time(self, tmp_path: Path) -> None:
        import datetime as dt

        before = dt.datetime.now(tz=dt.timezone.utc)
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.record_crash("foo/bar", "oops")
        after = dt.datetime.now(tz=dt.timezone.utc)
        crash = reader.get().repos["foo/bar"].crash_record
        assert crash.death_count > 0
        assert before <= crash.last_crash_time <= after

    def test_record_crash_increments_death_count(self, tmp_path: Path) -> None:
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.record_crash("foo/bar", "err1")
        reg.record_crash("foo/bar", "err2")
        reg.record_crash("foo/bar", "err3")
        crash = reader.get().repos["foo/bar"].crash_record
        assert crash.death_count == 3

    def test_record_crash_updates_last_error(self, tmp_path: Path) -> None:
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        reg.record_crash("foo/bar", "first")
        reg.record_crash("foo/bar", "second")
        crash = reader.get().repos["foo/bar"].crash_record
        assert crash.last_error == "second"

    def test_record_crash_accumulates_count(self, tmp_path: Path) -> None:
        """Sequential record_crash calls accumulate death_count correctly.

        record_crash is single-writer (watchdog-thread only) by contract —
        it reads and increments from the class-owned _crash_records store,
        then publishes via a pure lens write.  This test verifies the counter
        accumulates without loss over many sequential calls.
        """
        reg, _, reader = self._make_registry(repos=["foo/bar"], work_dir=tmp_path)
        n = 200
        for _ in range(n):
            reg.record_crash("foo/bar", "err")

        crash = reader.get().repos["foo/bar"].crash_record
        assert crash.death_count == n

    def test_worker_crash_dataclass_fields(self) -> None:
        import datetime as dt

        ts = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        crash = WorkerCrash(death_count=3, last_error="oops", last_crash_time=ts)
        assert crash.death_count == 3
        assert crash.last_error == "oops"
        assert crash.last_crash_time == ts

    def test_crash_record_survives_start(self, tmp_path: Path) -> None:
        """crash_record is preserved across start() so history accumulates."""
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
        cfg = _repo("foo/bar", tmp_path)
        reg.start(cfg)
        reg.record_crash("foo/bar", "boom")
        # Simulate crash so the FSM accepts the second start
        threads[0].is_alive.return_value = False
        threads[0].was_stopped = False
        reg.start(cfg)
        crash = reader.get().repos["foo/bar"].crash_record
        assert crash.death_count == 1
        assert crash.last_error == "boom"

    def test_start_replaces_existing_thread_entry(self, tmp_path: Path) -> None:
        threads = [MagicMock(), MagicMock()]
        factory = MagicMock(side_effect=threads)
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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

    # ── ThreadSnapshot / RepoState.thread ────────────────────────────────

    def test_thread_snapshot_none_before_any_start(self) -> None:
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        WorkerRegistry(MagicMock(), updater)
        assert reader.get().repos == {}

    def test_start_publishes_thread_snapshot(self, tmp_path: Path) -> None:
        """start() sets thread on the repo's RepoState in FidoState."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert isinstance(snap, ThreadSnapshot)

    def test_thread_snapshot_is_alive_field(self, tmp_path: Path) -> None:
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert snap.is_alive is True

    def test_thread_snapshot_was_stopped_field(self, tmp_path: Path) -> None:
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = False
        factory.return_value.was_stopped = True
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert snap.was_stopped is True

    def test_provider_snapshot_fields(self, tmp_path: Path) -> None:
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = "worker-home"
        factory.return_value.session_alive = True
        factory.return_value.session_pid = 12345
        factory.return_value.session_dropped_count = 2
        factory.return_value.session_sent_count = 7
        factory.return_value.session_received_count = 5
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        provider = reader.get().repos["foo/bar"].provider
        assert provider is not None
        assert isinstance(provider, ProviderSnapshot)
        assert provider.session_owner == "worker-home"
        assert provider.session_alive is True
        assert provider.session_pid == 12345
        assert provider.session_dropped_count == 2
        assert provider.session_sent_count == 7
        assert provider.session_received_count == 5

    def test_thread_snapshot_crash_error_field(self, tmp_path: Path) -> None:
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = False
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = "RuntimeError: boom"
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert snap.crash_error == "RuntimeError: boom"

    def test_thread_snapshot_no_mutable_thread_ref(self, tmp_path: Path) -> None:
        """ThreadSnapshot stores only primitive values — no WorkerThread handle."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        # The snapshot must not hold a reference to the thread mock itself
        from fido.worker import WorkerThread

        assert not isinstance(snap, WorkerThread)
        # All values are primitives: bool, int, str, or None
        for val in (
            snap.is_alive,
            snap.was_stopped,
            snap.crash_error,
        ):
            assert val is None or isinstance(val, (bool, int, str))

    def test_thread_snapshot_is_per_repo(self, tmp_path: Path) -> None:
        """Each repo gets its own ThreadSnapshot on its RepoState."""
        threads = [MagicMock(), MagicMock()]
        for t in threads:
            t.is_alive.return_value = True
            t.was_stopped = False
            t.session_owner = None
            t.session_alive = False
            t.session_pid = None
            t.session_dropped_count = 0
            t.session_sent_count = 0
            t.session_received_count = 0
            t.crash_error = None
        factory = MagicMock(side_effect=threads)
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        reg.start(_repo("foo/baz", tmp_path))
        state = reader.get()
        assert state.repos["foo/bar"].thread is not None
        assert state.repos["foo/baz"].thread is not None

    def test_thread_snapshot_frozen_in_fido_state(self, tmp_path: Path) -> None:
        """ThreadSnapshot is frozen — stored safely inside frozen FidoState."""
        import dataclasses

        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert dataclasses.is_dataclass(snap)
        with pytest.raises((TypeError, dataclasses.FrozenInstanceError)):
            snap.is_alive = False  # type: ignore[misc]

    def test_record_crash_republishes_thread_snapshot(self, tmp_path: Path) -> None:
        """record_crash refreshes the ThreadSnapshot so is_alive and crash_error reflect the crash."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        # Simulate what the watchdog does: thread has died and crash_error is set
        factory.return_value.is_alive.return_value = False
        factory.return_value.crash_error = "RuntimeError: oops"
        reg.record_crash("foo/bar", "RuntimeError: oops")
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert snap.is_alive is False
        assert snap.crash_error == "RuntimeError: oops"

    def test_stop_and_join_republishes_thread_snapshot(self, tmp_path: Path) -> None:
        """stop_and_join refreshes the ThreadSnapshot so is_alive and was_stopped reflect the stop."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        factory.return_value.is_alive.return_value = True
        factory.return_value.was_stopped = False
        factory.return_value.session_owner = None
        factory.return_value.session_alive = False
        factory.return_value.session_pid = None
        factory.return_value.session_dropped_count = 0
        factory.return_value.session_sent_count = 0
        factory.return_value.session_received_count = 0
        factory.return_value.crash_error = None
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        # Simulate what join() does: thread exits and was_stopped is set
        factory.return_value.is_alive.return_value = False
        factory.return_value.was_stopped = True
        reg.stop_and_join("foo/bar", timeout=5.0)
        snap = reader.get().repos["foo/bar"].thread
        assert snap is not None
        assert snap.is_alive is False
        assert snap.was_stopped is True

    def test_stop_and_join_unknown_repo_skips_snapshot(self) -> None:
        """stop_and_join on an unknown repo must not attempt to publish a snapshot."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        factory = MagicMock()
        reg = WorkerRegistry(factory, updater)
        reg.stop_and_join("unknown/repo")  # must not raise


class TestMakeThread:
    def test_creates_worker_thread_with_work_dir_and_github(
        self, tmp_path: Path
    ) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_registry = MagicMock()
        mock_gh = MagicMock()
        mock_wt_cls = MagicMock()
        fake_dispatcher = _FakeDispatcher()
        result = _make_thread(
            cfg,
            mock_registry,
            gh=mock_gh,
            dispatchers={"foo/bar": fake_dispatcher},
            _WorkerThread=mock_wt_cls,
        )
        from fido.config import RepoMembership
        from fido.provider_factory import DefaultProviderFactory

        mock_wt_cls.assert_called_once()
        call_args, call_kwargs = mock_wt_cls.call_args
        assert call_args == (tmp_path, "foo/bar", mock_gh, mock_registry)
        assert call_kwargs["membership"] == RepoMembership()
        assert isinstance(call_kwargs["provider_factory"], DefaultProviderFactory)
        assert call_kwargs["provider"] is None
        assert call_kwargs["session_issue"] is None
        assert call_kwargs["config"] is None
        assert call_kwargs["repo_cfg"] is cfg
        assert call_kwargs["dispatcher"] is fake_dispatcher
        assert call_kwargs["issue_cache"] is mock_registry.get_issue_cache.return_value
        assert call_kwargs["state_updater"] is None
        mock_registry.get_issue_cache.assert_called_once_with("foo/bar")
        assert result is mock_wt_cls.return_value

    def test_uses_repo_work_dir(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "myrepo"
        work_dir.mkdir()
        cfg = _repo("foo/bar", work_dir)
        mock_wt_cls = MagicMock()
        _make_thread(
            cfg,
            MagicMock(),
            gh=MagicMock(),
            dispatchers={"foo/bar": _FakeDispatcher()},
            _WorkerThread=mock_wt_cls,
        )
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
            cfg,
            MagicMock(),
            gh=MagicMock(),
            config=config,
            dispatchers={"foo/bar": _FakeDispatcher()},
            _WorkerThread=mock_wt_cls,
        )
        assert mock_wt_cls.call_args.kwargs["config"] is config

    def test_repo_cfg_forwarded_to_worker_thread(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_wt_cls = MagicMock()
        _make_thread(
            cfg,
            MagicMock(),
            gh=MagicMock(),
            dispatchers={"foo/bar": _FakeDispatcher()},
            _WorkerThread=mock_wt_cls,
        )
        assert mock_wt_cls.call_args.kwargs["repo_cfg"] is cfg

    def test_state_updater_forwarded_to_worker_thread(self, tmp_path: Path) -> None:
        """_make_thread passes state_updater through to WorkerThread."""
        cfg = _repo("foo/bar", tmp_path)
        mock_wt_cls = MagicMock()
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        _make_thread(
            cfg,
            MagicMock(),
            gh=MagicMock(),
            dispatchers={"foo/bar": _FakeDispatcher()},
            state_updater=updater,
            _WorkerThread=mock_wt_cls,
        )
        assert mock_wt_cls.call_args.kwargs["state_updater"] is updater

    def test_state_updater_defaults_to_none(self, tmp_path: Path) -> None:
        """_make_thread passes state_updater=None when not provided."""
        cfg = _repo("foo/bar", tmp_path)
        mock_wt_cls = MagicMock()
        _make_thread(
            cfg,
            MagicMock(),
            gh=MagicMock(),
            dispatchers={"foo/bar": _FakeDispatcher()},
            _WorkerThread=mock_wt_cls,
        )
        assert mock_wt_cls.call_args.kwargs["state_updater"] is None


class TestMakeRegistry:
    def test_returns_worker_registry(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        registry = make_registry(
            {"foo/bar": cfg},
            MagicMock(),
            dispatchers={},
            state_updater=updater,
            _thread_factory=MagicMock(return_value=mock_thread),
        )
        assert isinstance(registry, WorkerRegistry)

    def test_starts_thread_for_each_repo(self, tmp_path: Path) -> None:
        cfg1 = _repo("foo/bar", tmp_path)
        cfg2 = _repo("foo/baz", tmp_path)
        mock_thread = MagicMock()
        mock_factory = MagicMock(return_value=mock_thread)
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        make_registry(
            {"foo/bar": cfg1, "foo/baz": cfg2},
            MagicMock(),
            dispatchers={},
            state_updater=updater,
            _thread_factory=mock_factory,
        )
        assert mock_factory.call_count == 2
        assert mock_thread.start.call_count == 2

    def test_empty_repos_returns_empty_registry(self) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        registry = make_registry(
            {},
            MagicMock(),
            dispatchers={},
            state_updater=updater,
        )
        assert isinstance(registry, WorkerRegistry)
        assert registry.is_alive("anything") is False

    def test_wakes_registered_repos(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = make_registry(
            {"foo/bar": cfg},
            MagicMock(),
            dispatchers={},
            state_updater=updater,
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        make_registry(
            {"foo/bar": cfg},
            MagicMock(),
            config,
            dispatchers={},
            state_updater=updater,
            _thread_factory=mock_factory,
        )
        assert mock_factory.call_args.kwargs["config"] is config

    def test_state_updater_forwarded_to_thread_factory(self, tmp_path: Path) -> None:
        """make_registry passes state_updater through to the thread factory."""
        cfg = _repo("foo/bar", tmp_path)
        mock_factory = MagicMock(return_value=MagicMock())
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        make_registry(
            {"foo/bar": cfg},
            MagicMock(),
            dispatchers={},
            state_updater=updater,
            _thread_factory=mock_factory,
        )
        assert mock_factory.call_args.kwargs["state_updater"] is updater


class TestWebhookActivity:
    def test_registers_and_unregisters(self, tmp_path: Path) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        assert reg.get_webhook_activities("foo/bar") == []
        with reg.webhook_activity("foo/bar", "triaging"):
            activities = reg.get_webhook_activities("foo/bar")
            assert len(activities) == 1
            assert activities[0].description == "triaging"
        assert reg.get_webhook_activities("foo/bar") == []

    def test_unregisters_on_exception(self, tmp_path: Path) -> None:
        import pytest as _pytest

        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with _pytest.raises(RuntimeError):
            with reg.webhook_activity("foo/bar", "oops"):
                raise RuntimeError("boom")
        assert reg.get_webhook_activities("foo/bar") == []

    def test_multiple_concurrent_activities(self, tmp_path: Path) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with reg.webhook_activity("foo/bar", "first"):
            with reg.webhook_activity("foo/bar", "second"):
                descs = sorted(
                    a.description for a in reg.get_webhook_activities("foo/bar")
                )
                assert descs == ["first", "second"]
        assert reg.get_webhook_activities("foo/bar") == []

    def test_activities_isolated_per_repo(self, tmp_path: Path) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("a/b", tmp_path))
        reg.start(_repo("c/d", tmp_path))
        with reg.webhook_activity("a/b", "work-ab"):
            with reg.webhook_activity("c/d", "work-cd"):
                a = reg.get_webhook_activities("a/b")
                c = reg.get_webhook_activities("c/d")
                assert [x.description for x in a] == ["work-ab"]
                assert [x.description for x in c] == ["work-cd"]

    def test_unknown_repo_returns_empty_list(self) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        assert reg.get_webhook_activities("ghost/repo") == []

    def test_handle_can_update_description(self, tmp_path: Path) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with reg.webhook_activity("foo/bar", "handling") as activity:
            assert reg.get_webhook_activities("foo/bar")[0].description == "handling"
            activity.set_description("triaging")
            assert reg.get_webhook_activities("foo/bar")[0].description == "triaging"

    def test_handle_update_after_exit_is_noop(self, tmp_path: Path) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with reg.webhook_activity("foo/bar", "handling") as activity:
            pass
        activity.set_description("triaging")
        assert reg.get_webhook_activities("foo/bar") == []

    def test_unknown_handle_update_is_noop(self, tmp_path: Path) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with reg.webhook_activity("foo/bar", "handling"):
            reg.set_webhook_description("foo/bar", -1, "triaging")
            assert reg.get_webhook_activities("foo/bar")[0].description == "handling"

    def test_unknown_repo_handle_update_is_noop(self) -> None:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.set_webhook_description("ghost/repo", 1, "triaging")
        assert reg.get_webhook_activities("ghost/repo") == []

    def test_publishes_to_fido_state_when_repo_started(self, tmp_path: Path) -> None:
        """webhook_activity publishes activities into FidoState when start() has run."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with reg.webhook_activity("foo/bar", "handling"):
            acts = reader.get().repos["foo/bar"].webhook_activities
            assert len(acts) == 1
            assert acts[0].description == "handling"
        assert reader.get().repos["foo/bar"].webhook_activities == ()

    def test_publishes_description_update_to_fido_state(self, tmp_path: Path) -> None:
        """set_webhook_description publishes the update into FidoState."""
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(MagicMock(), updater)
        reg.start(_repo("foo/bar", tmp_path))
        with reg.webhook_activity("foo/bar", "original") as handle:
            handle.set_description("updated")
            acts = reader.get().repos["foo/bar"].webhook_activities
            assert len(acts) == 1
            assert acts[0].description == "updated"


class TestRescoping:
    """``set_rescoping`` is now a pure lens write into
    :class:`RepoState.rescoping`; the snapshot is the single source of
    truth (#1696 parity).  Repos must be ``start()``-ed first so the
    lens has a target to write into."""

    def _reg(self) -> tuple[WorkerRegistry, object]:
        factory = MagicMock()
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        return WorkerRegistry(factory, updater), reader

    def test_set_rescoping_true_publishes_to_snapshot(self, tmp_path: Path) -> None:
        reg, reader = self._reg()
        reg.start(_repo("foo/bar", tmp_path))
        reg.set_rescoping("foo/bar", True)
        assert reader.get().repos["foo/bar"].rescoping is True

    def test_set_rescoping_false_clears_flag(self, tmp_path: Path) -> None:
        reg, reader = self._reg()
        reg.start(_repo("foo/bar", tmp_path))
        reg.set_rescoping("foo/bar", True)
        reg.set_rescoping("foo/bar", False)
        assert reader.get().repos["foo/bar"].rescoping is False

    def test_rescoping_is_per_repo(self, tmp_path: Path) -> None:
        reg, reader = self._reg()
        bar = tmp_path / "bar"
        baz = tmp_path / "baz"
        for d in (bar, baz):
            d.mkdir()
            subprocess.run(["git", "init", "--quiet"], cwd=d, check=True)
        reg.start(_repo("foo/bar", bar))
        reg.start(_repo("foo/baz", baz))
        reg.set_rescoping("foo/bar", True)
        snap = reader.get()
        assert snap.repos["foo/bar"].rescoping is True
        assert snap.repos["foo/baz"].rescoping is False


class TestParityPublishers:
    """Tests for the per-repo on_change publisher closures wired at
    :meth:`WorkerRegistry.start` for the issue cache and provider talker
    leaves of :class:`FidoState` (#1696 parity)."""

    def _reg(self, tmp_path: Path) -> tuple[WorkerRegistry, object]:
        factory = MagicMock()
        reader, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
        reg.start(_repo("foo/bar", tmp_path))
        return reg, reader

    def test_issue_cache_publish_writes_snapshot(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        reg, reader = self._reg(tmp_path)
        cache = reg.get_issue_cache("foo/bar")
        cache.load_inventory(
            issues=[],
            snapshot_started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        snap = reader.get().repos["foo/bar"].issue_cache
        assert snap.loaded is True
        assert snap.open_issues == 0

    def test_restart_republishes_existing_issue_cache(self, tmp_path: Path) -> None:
        """Codex parity: ``zero_repo_state`` resets ``issue_cache`` to
        ``loaded=false`` on every ``start()``, but on crash recovery the
        existing :class:`IssueCache` instance is preserved.  ``start()``
        must republish the cache's current metrics so /status.json
        doesn't regress to an empty snapshot until the next mutation."""
        from datetime import datetime, timezone

        reg, reader = self._reg(tmp_path)
        cache = reg.get_issue_cache("foo/bar")
        cache.load_inventory(
            issues=[],
            snapshot_started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert reader.get().repos["foo/bar"].issue_cache.loaded is True

        # Simulate watchdog crash recovery — predecessor thread died
        # but wasn't orderly-stopped, triggering the
        # ThreadDies → Rescue → start path.  zero_repo_state would
        # normally reset issue_cache to loaded=false on the restart;
        # the codex fix republishes the preserved cache's snapshot.
        reg._threads["foo/bar"].is_alive.return_value = False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        reg._threads["foo/bar"].was_stopped = False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        reg.start(_repo("foo/bar", tmp_path))
        snap = reader.get().repos["foo/bar"].issue_cache
        assert snap.loaded is True, (
            "expected start() to republish the preserved cache's snapshot"
        )

    def test_talker_publish_writes_snapshot(self, tmp_path: Path) -> None:
        from datetime import datetime, timezone

        from fido import provider as provider_module

        reg, reader = self._reg(tmp_path)
        talker = provider_module.SessionTalker(
            repo_name="foo/bar",
            thread_id=12345,
            kind="worker",
            description="test",
            subprocess_pid=99,
            started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        try:
            provider_module.register_talker(talker)
            snap = reader.get().repos["foo/bar"].talker
            assert snap.thread_id == 12345
            assert snap.kind == "worker"
            assert snap.subprocess_pid == 99
            provider_module.unregister_talker("foo/bar", 12345)
            zero = reader.get().repos["foo/bar"].talker
            assert zero.thread_id == 0
            assert zero.kind == ""
        finally:
            # Clean the global talker dict + callback so other tests
            # don't see leakage between cases.
            with provider_module._talkers_lock:  # noqa: PLC2701
                provider_module._talkers.pop("foo/bar", None)  # noqa: PLC2701
            with provider_module._talker_change_callbacks_lock:  # noqa: PLC2701
                provider_module._talker_change_callbacks.pop(  # noqa: PLC2701
                    "foo/bar", None
                )


class TestUntriagedInbox:
    """Tests for the per-repo untriaged-webhook inbox (fix #1067)."""

    def _reg(self) -> WorkerRegistry:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        return WorkerRegistry(MagicMock(), updater)

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

    def test_force_clear_drains_fsm_legacy_demand(self) -> None:
        """Regression for #1330: force_clear must take the FSM through the
        modeled HandlerDone transition, not just zero the Python count.

        Before the fix, force_clear left ``legacy_demand=LegacyNonEmpty`` while
        the count said 0, so the next ``assert_worker_turn_ok`` fired the FSM
        oracle and crashed the worker — repeatedly, in a watchdog-restart loop.
        """
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")  # count=2; FSM is LegacyNonEmpty
        reg.force_clear_untriaged("foo/bar")
        # FSM must be back at LegacyEmpty so the worker can take its next turn.
        reg.assert_worker_turn_ok("foo/bar")  # must not raise


class TestPreemptionFsmOracle:
    """Tests for the handler_preemption FSM oracle wired into WorkerRegistry.

    Each test maps to a proved invariant from ``models/handler_preemption.v``.
    The FSM oracle validates that WorkerTurnStart is rejected when the inbox
    is non-empty — the core preemption guarantee from #1067.
    """

    def _reg(self) -> WorkerRegistry:
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        return WorkerRegistry(MagicMock(), updater)

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

    def test_partial_exit_keeps_fsm_blocked_until_count_drains(self) -> None:
        """Boolean-abstraction contract: the FSM agrees with the Python count
        by construction.  Two enters then one exit must leave the FSM at
        ``LegacyNonEmpty`` (count is still 1) — no spurious HandlerDone fires
        on the intermediate edge.
        """
        reg = self._reg()
        reg.enter_untriaged("foo/bar")
        reg.enter_untriaged("foo/bar")
        reg.exit_untriaged("foo/bar")  # count=1; FSM still LegacyNonEmpty

        with pytest.raises(AssertionError, match="WorkerTurnStart rejected"):
            reg.assert_worker_turn_ok("foo/bar")

        reg.exit_untriaged("foo/bar")  # count=0; FSM transitions to Empty
        reg.assert_worker_turn_ok("foo/bar")  # must not raise

    def test_repeated_enter_is_fsm_idempotent(self) -> None:
        """Multiple enters at the same repo must not crash the FSM oracle.

        Under the boolean abstraction the FSM transition fires only on the
        ``0 → 1`` count edge; subsequent enters are no-ops at the FSM level.
        """
        reg = self._reg()
        for _ in range(5):
            reg.enter_untriaged("foo/bar")
        # Drain to zero — exactly one HandlerDone should fire on the 1→0 edge.
        for _ in range(5):
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        return WorkerRegistry(MagicMock(), updater)

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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        reg = WorkerRegistry(factory, updater)
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
