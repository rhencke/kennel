"""Tests for kennel.worker — WorkerContext, lock acquisition, git context."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.worker import (
    LockHeld,
    RepoContext,
    Worker,
    WorkerContext,
    run,
)


class TestResolveGitDir:
    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def test_returns_path(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/some/repo/.git\n")
            result = self._make_worker(tmp_path).resolve_git_dir()
        assert result == Path("/some/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  /a/b/.git  \n")
            result = self._make_worker(tmp_path).resolve_git_dir()
        assert result == Path("/a/b/.git")

    def test_calls_correct_command(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/a/.git")
            self._make_worker(tmp_path).resolve_git_dir()
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_propagates_called_process_error(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")
            with pytest.raises(subprocess.CalledProcessError):
                self._make_worker(tmp_path).resolve_git_dir()


class TestAcquireLock:
    def test_returns_open_fd(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = Worker.acquire_lock(fido_dir)
        assert not fd.closed
        fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "nested" / "fido"
        fd = Worker.acquire_lock(fido_dir)
        assert fido_dir.is_dir()
        fd.close()

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = Worker.acquire_lock(fido_dir)
        assert (fido_dir / "lock").exists()
        fd.close()

    def test_raises_lock_held_when_already_locked(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = Worker.acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld):
                Worker.acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_lock_held_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = Worker.acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld, match="another fido"):
                Worker.acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_reacquirable_after_release(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = Worker.acquire_lock(fido_dir)
        fd1.close()
        fd2 = Worker.acquire_lock(fido_dir)
        assert not fd2.closed
        fd2.close()


class TestCreateContext:
    def test_returns_worker_context(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        worker = Worker(tmp_path, MagicMock())
        with patch.object(worker, "resolve_git_dir", return_value=git_dir):
            ctx = worker.create_context()
        assert isinstance(ctx, WorkerContext)
        assert ctx.work_dir == tmp_path
        assert ctx.git_dir == git_dir
        assert ctx.fido_dir == git_dir / "fido"
        ctx.lock_fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        worker = Worker(tmp_path, MagicMock())
        with patch.object(worker, "resolve_git_dir", return_value=git_dir):
            ctx = worker.create_context()
        assert ctx.fido_dir.is_dir()
        ctx.lock_fd.close()

    def test_propagates_lock_held(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        fido_dir = git_dir / "fido"
        worker = Worker(tmp_path, MagicMock())
        with patch.object(worker, "resolve_git_dir", return_value=git_dir):
            fd1 = Worker.acquire_lock(fido_dir)
            try:
                with pytest.raises(LockHeld):
                    worker.create_context()
            finally:
                fd1.close()


class TestRepoContext:
    def test_fields(self) -> None:
        ctx = RepoContext(
            repo="alice/myrepo",
            owner="alice",
            repo_name="myrepo",
            gh_user="bot",
            default_branch="main",
        )
        assert ctx.repo == "alice/myrepo"
        assert ctx.owner == "alice"
        assert ctx.repo_name == "myrepo"
        assert ctx.gh_user == "bot"
        assert ctx.default_branch == "main"


class TestWorker:
    def _make_gh(
        self,
        repo: str = "owner/myrepo",
        user: str = "fido-bot",
        branch: str = "main",
    ) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = repo
        gh.get_user.return_value = user
        gh.get_default_branch.return_value = branch
        return gh

    # --- discover_repo_context ---

    def test_discover_returns_repo_context(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        result = Worker(tmp_path, gh).discover_repo_context()
        assert isinstance(result, RepoContext)

    def test_discover_repo_field(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.repo == "alice/proj"

    def test_discover_owner_parsed(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.owner == "alice"

    def test_discover_repo_name_parsed(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.repo_name == "proj"

    def test_discover_gh_user(self, tmp_path: Path) -> None:
        gh = self._make_gh(user="fido")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.gh_user == "fido"

    def test_discover_default_branch(self, tmp_path: Path) -> None:
        gh = self._make_gh(branch="develop")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.default_branch == "develop"

    def test_discover_passes_cwd_to_get_repo_info(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        Worker(tmp_path, gh).discover_repo_context()
        gh.get_repo_info.assert_called_once_with(cwd=tmp_path)

    def test_discover_passes_cwd_to_get_default_branch(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        Worker(tmp_path, gh).discover_repo_context()
        gh.get_default_branch.assert_called_once_with(cwd=tmp_path)

    def test_discover_splits_on_first_slash_only(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="org/repo")
        result = Worker(tmp_path, gh).discover_repo_context()
        assert result.owner == "org"
        assert result.repo_name == "repo"

    # --- set_status ---

    def test_set_status_calls_set_user_status_on_success(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status", return_value="🐕\nwriting tests"
            ),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("writing tests")
        gh.set_user_status.assert_called_once_with("writing tests", "🐕", busy=True)

    def test_set_status_busy_false_forwarded(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="😴\nnapping"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("napping", busy=False)
        gh.set_user_status.assert_called_once_with("napping", "😴", busy=False)

    def test_set_status_skips_when_claude_returns_empty(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value=""),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("idle")
        gh.set_user_status.assert_not_called()

    def test_set_status_skips_when_only_one_line(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("idle")
        gh.set_user_status.assert_not_called()

    def test_set_status_text_truncated_to_80_chars(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        long_text = "x" * 100
        with (
            patch(
                "kennel.worker.claude.generate_status",
                return_value=f"🐕\n{long_text}",
            ),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("something")
        called_text = gh.set_user_status.call_args[0][0]
        assert len(called_text) == 80

    def test_set_status_logs_warning_on_empty_response(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value=""),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.WARNING, logger="kennel"):
                Worker(tmp_path, gh).set_status("idle")
        assert "empty" in caplog.text

    def test_set_status_logs_warning_on_single_line(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.WARNING, logger="kennel"):
                Worker(tmp_path, gh).set_status("idle")
        assert "expected 2 lines" in caplog.text

    def test_set_status_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="🐕\nfetching"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.INFO, logger="kennel"):
                Worker(tmp_path, gh).set_status("fetching")
        assert "set_status" in caplog.text

    def test_set_status_falls_back_to_empty_persona_on_oserror(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        missing_dir = tmp_path / "no_such_dir"
        with (
            patch(
                "kennel.worker.claude.generate_status", return_value="🐕\nworking"
            ) as mock_gen,
            patch("kennel.worker._sub_dir", return_value=missing_dir),
        ):
            Worker(tmp_path, gh).set_status("working")
        # persona file missing — generate_status still called with empty persona
        prompt_arg = mock_gen.call_args[1]["prompt"]
        assert "What you're doing right now: working" in prompt_arg

    def test_set_status_passes_system_prompt_to_generate_status(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status", return_value="🐕\nworking"
            ) as mock_gen,
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("working")
        assert mock_gen.call_args[1]["system_prompt"] is not None

    # --- get_current_issue ---

    def _make_issue_gh(self, state: str = "OPEN") -> MagicMock:
        gh = MagicMock()
        gh.view_issue.return_value = {"state": state, "title": "Test", "body": ""}
        return gh

    def test_get_issue_returns_none_when_no_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        gh = self._make_issue_gh()
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") is None

    def test_get_issue_returns_issue_number_when_open(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 7})
        gh = self._make_issue_gh(state="OPEN")
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") == 7

    def test_get_issue_returns_int_type(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 7})
        gh = self._make_issue_gh(state="OPEN")
        result = Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert isinstance(result, int)

    def test_get_issue_returns_none_when_closed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 4})
        gh = self._make_issue_gh(state="CLOSED")
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") is None

    def test_get_issue_clears_state_when_closed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 4})
        gh = self._make_issue_gh(state="CLOSED")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert Worker.load_state(fido_dir) == {}

    def test_get_issue_does_not_call_view_issue_when_no_state(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        gh = self._make_issue_gh()
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        gh.view_issue.assert_not_called()

    def test_get_issue_calls_view_issue_with_correct_args(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 12})
        gh = self._make_issue_gh(state="OPEN")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "alice/proj")
        gh.view_issue.assert_called_once_with("alice/proj", 12)

    def test_get_issue_logs_info_when_closed(self, tmp_path: Path, caplog) -> None:
        import logging

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 9})
        gh = self._make_issue_gh(state="CLOSED")
        with caplog.at_level(logging.INFO, logger="kennel"):
            Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert "advancing" in caplog.text

    def test_get_issue_state_preserved_when_open(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 5})
        gh = self._make_issue_gh(state="OPEN")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert Worker.load_state(fido_dir) == {"issue": 5}

    # --- run ---

    def _make_mock_ctx(self, tmp_path: Path) -> MagicMock:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        mock_ctx.fido_dir = tmp_path / ".git" / "fido"
        return mock_ctx

    def _make_mock_repo_ctx(self) -> MagicMock:
        repo_ctx = MagicMock(spec=RepoContext)
        repo_ctx.repo = "owner/repo"
        repo_ctx.gh_user = "fido-bot"
        repo_ctx.default_branch = "main"
        return repo_ctx

    def test_run_returns_2_when_lock_held(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with patch.object(worker, "create_context", side_effect=LockHeld("held")):
            assert worker.run() == 2

    def _run_patches(self, worker: Worker, mock_ctx: MagicMock) -> list:
        """Return a list of common patches needed to run Worker.run() in tests.

        Patches create_context, discover_repo_context, setup_hooks, teardown_hooks,
        get_current_issue (→ None), and find_next_issue (→ None) so the loop exits
        cleanly without hitting GitHub or Claude.
        """
        return [
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ]

    def test_run_returns_0_on_success(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            assert worker.run() == 0

    def test_run_logs_warning_on_lock_held(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with patch.object(worker, "create_context", side_effect=LockHeld("held")):
            with caplog.at_level(logging.WARNING, logger="kennel"):
                worker.run()
        assert "another fido" in caplog.text

    def test_run_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                worker.run()
        assert "worker started" in caplog.text

    def test_run_logs_repo_info(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                worker.run()
        assert "owner/repo" in caplog.text

    def test_run_teardown_called_even_on_exception(self, tmp_path: Path) -> None:
        """teardown_hooks must be called even if the main loop raises."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_teardown = MagicMock()
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(
                worker, "setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch.object(worker, "teardown_hooks", mock_teardown),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_teardown.assert_called_once()

    def test_run_setup_hooks_called_with_fido_dir(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_setup = MagicMock(return_value=("c", "s"))
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", mock_setup),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_setup.assert_called_once_with(mock_ctx.fido_dir)

    def test_run_calls_get_current_issue(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        mock_get_issue = MagicMock(return_value=None)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", mock_get_issue),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            worker.run()
        mock_get_issue.assert_called_once_with(mock_ctx.fido_dir, "owner/repo")

    def test_run_calls_find_next_issue_when_no_current(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        mock_find = MagicMock(return_value=None)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", mock_find),
        ):
            worker.run()
        mock_find.assert_called_once_with(mock_ctx.fido_dir, repo_ctx)

    def test_run_skips_find_next_when_current_issue_exists(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix bug", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_find = MagicMock(return_value=None)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "find_next_issue", mock_find),
            patch.object(worker, "post_pickup_comment"),
        ):
            worker.run()
        mock_find.assert_not_called()

    def test_run_returns_0_when_no_issue(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=None),
            patch.object(worker, "find_next_issue", return_value=None),
        ):
            assert worker.run() == 0

    def test_run_calls_post_pickup_comment_when_issue_found(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {
            "title": "Test issue",
            "body": "",
            "state": "OPEN",
        }
        worker = Worker(tmp_path, gh)
        mock_pickup = MagicMock()
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=5),
            patch.object(worker, "post_pickup_comment", mock_pickup),
        ):
            worker.run()
        mock_pickup.assert_called_once_with("owner/repo", 5, "Test issue", "fido-bot")

    def test_run_views_issue_for_title(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {
            "title": "Some title",
            "body": "",
            "state": "OPEN",
        }
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=3),
            patch.object(worker, "post_pickup_comment"),
        ):
            worker.run()
        gh.view_issue.assert_called_once_with("owner/repo", 3)


class TestWorkerFindNextIssue:
    """Tests for Worker.find_next_issue."""

    def _make_worker(self, tmp_path: Path) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _make_repo_ctx(
        self,
        owner: str = "alice",
        repo_name: str = "proj",
        repo: str = "alice/proj",
        gh_user: str = "fido-bot",
    ) -> RepoContext:
        return RepoContext(
            repo=repo,
            owner=owner,
            repo_name=repo_name,
            gh_user=gh_user,
            default_branch="main",
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "fido"
        d.mkdir()
        return d

    def test_returns_none_when_no_issues(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result is None

    def test_returns_issue_number_when_eligible_no_subissues(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {"number": 42, "title": "Do the thing", "subIssues": {"nodes": []}}
        ]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 42

    def test_returns_issue_number_when_all_subissues_closed(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {
                "number": 10,
                "title": "Parent task",
                "subIssues": {"nodes": [{"state": "CLOSED"}, {"state": "CLOSED"}]},
            }
        ]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 10

    def test_skips_issue_with_open_subissue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {
                "number": 3,
                "title": "Blocked",
                "subIssues": {"nodes": [{"state": "OPEN"}]},
            }
        ]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result is None

    def test_picks_first_eligible_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {
                "number": 1,
                "title": "Blocked",
                "subIssues": {"nodes": [{"state": "OPEN"}]},
            },
            {
                "number": 2,
                "title": "Ready",
                "subIssues": {"nodes": []},
            },
            {
                "number": 3,
                "title": "Also ready",
                "subIssues": {"nodes": []},
            },
        ]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result == 2

    def test_saves_state_when_issue_found(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {"number": 7, "title": "Fetch!", "subIssues": {"nodes": []}}
        ]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert Worker.load_state(fido_dir) == {"issue": 7}

    def test_does_not_save_state_when_no_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert Worker.load_state(fido_dir) == {}

    def test_calls_set_status_with_issue_info_when_found(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {"number": 5, "title": "Add tests", "subIssues": {"nodes": []}}
        ]
        fido_dir = self._fido_dir(tmp_path)
        mock_status = MagicMock()
        with patch.object(worker, "set_status", mock_status):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        mock_status.assert_called_once_with("Picking up issue #5: Add tests")

    def test_calls_set_status_done_when_no_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        mock_status = MagicMock()
        with patch.object(worker, "set_status", mock_status):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        mock_status.assert_called_once_with("All done — no issues to fetch", busy=False)

    def test_passes_correct_args_to_find_issues(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        repo_ctx = self._make_repo_ctx(owner="org", repo_name="myrepo", gh_user="bot")
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, repo_ctx)
        gh.find_issues.assert_called_once_with("org", "myrepo", "bot")

    def test_logs_info_when_starting_issue(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {"number": 9, "title": "Chase squirrel", "subIssues": {"nodes": []}}
        ]
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert "9" in caplog.text

    def test_logs_info_when_no_eligible_issues(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert "no eligible" in caplog.text

    def test_mixed_closed_and_open_subissues_skips(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = [
            {
                "number": 11,
                "title": "Partial",
                "subIssues": {"nodes": [{"state": "CLOSED"}, {"state": "OPEN"}]},
            }
        ]
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            result = worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert result is None


class TestWorkerPostPickupComment:
    """Tests for Worker.post_pickup_comment."""

    def _make_worker(self, tmp_path: Path) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def test_skips_when_already_commented(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = [
            {"user": {"login": "fido-bot"}, "body": "Woof!"}
        ]
        worker.post_pickup_comment("owner/repo", 1, "Fix bug", "fido-bot")
        gh.comment_issue.assert_not_called()

    def test_posts_comment_when_no_previous_comment(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = [
            {"user": {"login": "other-user"}, "body": "Hi"}
        ]
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch("kennel.worker.claude.generate_reply", return_value="Woof! On it!"),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            worker.post_pickup_comment("owner/repo", 1, "Fix bug", "fido-bot")
        gh.comment_issue.assert_called_once_with("owner/repo", 1, "Woof! On it!")

    def test_posts_comment_when_no_existing_comments(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = []
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch("kennel.worker.claude.generate_reply", return_value="I am on it!"),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            worker.post_pickup_comment("owner/repo", 3, "Some task", "fido-bot")
        gh.comment_issue.assert_called_once()

    def test_falls_back_to_plain_text_when_claude_returns_empty(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = []
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch("kennel.worker.claude.generate_reply", return_value=""),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            worker.post_pickup_comment("owner/repo", 5, "A task", "fido-bot")
        gh.comment_issue.assert_called_once_with(
            "owner/repo", 5, "Picking up issue: A task"
        )

    def test_uses_persona_from_sub_dir(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = []
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch(
                "kennel.worker.claude.generate_reply", return_value="Fetched!"
            ) as mock_gen,
        ):
            (tmp_path / "persona.md").write_text("I am a very good dog.")
            worker.post_pickup_comment("owner/repo", 2, "Some work", "fido-bot")
        prompt_arg = mock_gen.call_args[0][0]
        assert "I am a very good dog." in prompt_arg

    def test_falls_back_to_empty_persona_on_oserror(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = []
        missing = tmp_path / "no_such_dir"
        with (
            patch("kennel.worker._sub_dir", return_value=missing),
            patch(
                "kennel.worker.claude.generate_reply", return_value="On it!"
            ) as mock_gen,
        ):
            worker.post_pickup_comment("owner/repo", 2, "Work item", "fido-bot")
        prompt_arg = mock_gen.call_args[0][0]
        assert "Picking up issue: Work item" in prompt_arg

    def test_prompt_includes_issue_title(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = []
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch(
                "kennel.worker.claude.generate_reply", return_value="On it!"
            ) as mock_gen,
        ):
            (tmp_path / "persona.md").write_text("")
            worker.post_pickup_comment("owner/repo", 4, "Refactor auth", "fido-bot")
        prompt_arg = mock_gen.call_args[0][0]
        assert "Refactor auth" in prompt_arg

    def test_checks_comments_for_correct_repo_and_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = []
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch("kennel.worker.claude.generate_reply", return_value="Arf!"),
        ):
            (tmp_path / "persona.md").write_text("")
            worker.post_pickup_comment("org/myrepo", 99, "Title", "fido-bot")
        gh.get_issue_comments.assert_called_once_with("org/myrepo", 99)

    def test_logs_info_when_skipping(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_issue_comments.return_value = [
            {"user": {"login": "fido-bot"}, "body": "Woof!"}
        ]
        with (
            patch("kennel.worker._sub_dir", return_value=tmp_path),
            patch("kennel.worker.claude.generate_reply", return_value="Woof!"),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                worker.post_pickup_comment("owner/repo", 7, "Title", "fido-bot")
        assert "already exists" in caplog.text


class TestRun:
    def test_creates_worker_and_calls_run(self, tmp_path: Path) -> None:
        mock_worker = MagicMock()
        mock_worker.run.return_value = 0
        with (
            patch("kennel.worker.GitHub") as mock_gh_cls,
            patch("kennel.worker.Worker", return_value=mock_worker) as mock_worker_cls,
        ):
            result = run(tmp_path)
        mock_worker_cls.assert_called_once_with(tmp_path, mock_gh_cls.return_value)
        mock_worker.run.assert_called_once()
        assert result == 0

    def test_returns_worker_run_result(self, tmp_path: Path) -> None:
        mock_worker = MagicMock()
        mock_worker.run.return_value = 2
        with (
            patch("kennel.worker.GitHub"),
            patch("kennel.worker.Worker", return_value=mock_worker),
        ):
            assert run(tmp_path) == 2


class TestCreateCompactScript:
    def test_creates_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        assert script.exists()

    def test_returns_path_in_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        assert script == fido_dir / "compact.sh"

    def test_script_is_executable(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        assert script.stat().st_mode & 0o111  # any execute bit

    def test_script_has_shebang(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        assert script.read_text().startswith("#!/usr/bin/env bash\n")

    def test_script_references_sub_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        from kennel.worker import _sub_dir

        assert str(_sub_dir()) in script.read_text()

    def test_script_contains_post_compact_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        assert "PostCompact" in script.read_text()

    def test_script_contains_md_glob(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = Worker.create_compact_script(fido_dir)
        assert "*.md" in script.read_text()


class TestSetupHooks:
    def test_returns_compact_and_sync_cmds(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, sync_cmd = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert compact_cmd.startswith("bash ")
        assert sync_cmd.startswith("bash ")

    def test_compact_cmd_references_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, _ = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert "compact.sh" in compact_cmd

    def test_sync_cmd_references_sync_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        _, sync_cmd = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert "sync-tasks.sh" in sync_cmd

    def test_sync_cmd_includes_work_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        _, sync_cmd = Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert str(tmp_path) in sync_cmd

    def test_creates_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        assert (fido_dir / "compact.sh").exists()

    def test_adds_hooks_to_settings(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        settings = tmp_path / ".claude" / "settings.local.json"
        assert settings.exists()
        cfg = json.loads(settings.read_text())
        assert "hooks" in cfg

    def test_gitexcludes_settings(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        Worker(tmp_path, MagicMock()).setup_hooks(fido_dir)
        exclude = tmp_path / ".git" / "info" / "exclude"
        assert ".claude/settings.local.json" in exclude.read_text()


class TestTeardownHooks:
    def test_removes_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        worker = Worker(tmp_path, MagicMock())
        compact_cmd, sync_cmd = worker.setup_hooks(fido_dir)
        worker.teardown_hooks(fido_dir, compact_cmd, sync_cmd)
        assert not (fido_dir / "compact.sh").exists()

    def test_removes_hooks_from_settings(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        worker = Worker(tmp_path, MagicMock())
        compact_cmd, sync_cmd = worker.setup_hooks(fido_dir)
        worker.teardown_hooks(fido_dir, compact_cmd, sync_cmd)
        settings = tmp_path / ".claude" / "settings.local.json"
        cfg = json.loads(settings.read_text())
        assert "hooks" not in cfg

    def test_noop_when_compact_script_missing(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # Should not raise even when compact.sh does not exist
        Worker(tmp_path, MagicMock()).teardown_hooks(
            fido_dir, "bash /x/compact.sh", "bash sync.sh &"
        )

    def test_noop_when_settings_missing(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # No settings file created — should not raise
        Worker(tmp_path, MagicMock()).teardown_hooks(
            fido_dir, "bash /x/compact.sh", "bash sync.sh &"
        )


class TestLoadState:
    def test_returns_empty_dict_when_absent(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        assert Worker.load_state(fido_dir) == {}

    def test_returns_state_when_present(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 42}))
        assert Worker.load_state(fido_dir) == {"issue": 42}

    def test_returns_dict_with_arbitrary_keys(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 7, "extra": "val"}))
        result = Worker.load_state(fido_dir)
        assert result["issue"] == 7
        assert result["extra"] == "val"


class TestSaveState:
    def test_creates_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 5})
        assert (fido_dir / "state.json").exists()

    def test_roundtrips_with_load_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 99})
        assert Worker.load_state(fido_dir) == {"issue": 99}

    def test_overwrites_existing_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 1})
        Worker.save_state(fido_dir, {"issue": 2})
        assert Worker.load_state(fido_dir) == {"issue": 2}


class TestClearState:
    def test_removes_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 3})
        Worker.clear_state(fido_dir)
        assert not (fido_dir / "state.json").exists()

    def test_noop_when_absent(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # Should not raise
        Worker.clear_state(fido_dir)

    def test_load_returns_empty_after_clear(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        Worker.save_state(fido_dir, {"issue": 10})
        Worker.clear_state(fido_dir)
        assert Worker.load_state(fido_dir) == {}
