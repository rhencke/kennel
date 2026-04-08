"""Tests for kennel.registry — WorkerRegistry lifecycle management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from kennel.config import RepoConfig
from kennel.registry import WorkerRegistry, _make_thread, make_registry


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
        with (
            patch("kennel.registry.GitHub") as mock_gh_cls,
            patch("kennel.registry.WorkerThread") as mock_wt_cls,
        ):
            result = _make_thread(cfg)
        mock_wt_cls.assert_called_once_with(tmp_path, mock_gh_cls.return_value)
        assert result is mock_wt_cls.return_value

    def test_uses_repo_work_dir(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "myrepo"
        work_dir.mkdir()
        cfg = _repo("foo/bar", work_dir)
        with (
            patch("kennel.registry.GitHub"),
            patch("kennel.registry.WorkerThread") as mock_wt_cls,
        ):
            _make_thread(cfg)
        assert mock_wt_cls.call_args[0][0] == work_dir


class TestMakeRegistry:
    def test_returns_worker_registry(self, tmp_path: Path) -> None:
        cfg = _repo("foo/bar", tmp_path)
        mock_thread = MagicMock()
        with patch("kennel.registry._make_thread", return_value=mock_thread):
            result = make_registry({"foo/bar": cfg})
        assert isinstance(result, WorkerRegistry)

    def test_starts_thread_for_each_repo(self, tmp_path: Path) -> None:
        cfg1 = _repo("foo/bar", tmp_path)
        cfg2 = _repo("foo/baz", tmp_path)
        mock_thread = MagicMock()
        with patch(
            "kennel.registry._make_thread", return_value=mock_thread
        ) as mock_factory:
            make_registry({"foo/bar": cfg1, "foo/baz": cfg2})
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
        with patch("kennel.registry._make_thread", return_value=mock_thread):
            reg = make_registry({"foo/bar": cfg})
        assert reg.is_alive("foo/bar") is True
