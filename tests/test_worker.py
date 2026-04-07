"""Tests for kennel.worker — WorkerContext, lock acquisition, git context."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from kennel.worker import (
    LockHeld,
    RepoContext,
    Worker,
    WorkerContext,
    _sanitize_slug,
    acquire_lock,
    build_prompt,
    claude_run,
    claude_start,
    clear_state,
    create_compact_script,
    load_state,
    run,
    save_state,
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
        fd = acquire_lock(fido_dir)
        assert not fd.closed
        fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "nested" / "fido"
        fd = acquire_lock(fido_dir)
        assert fido_dir.is_dir()
        fd.close()

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        assert (fido_dir / "lock").exists()
        fd.close()

    def test_raises_lock_held_when_already_locked(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld):
                acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_lock_held_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld, match="another fido"):
                acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_reacquirable_after_release(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        fd1.close()
        fd2 = acquire_lock(fido_dir)
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
            fd1 = acquire_lock(fido_dir)
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
        gh.get_pr.return_value = {"body": ""}
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
        save_state(fido_dir, {"issue": 7})
        gh = self._make_issue_gh(state="OPEN")
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") == 7

    def test_get_issue_returns_int_type(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 7})
        gh = self._make_issue_gh(state="OPEN")
        result = Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert isinstance(result, int)

    def test_get_issue_returns_none_when_closed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 4})
        gh = self._make_issue_gh(state="CLOSED")
        assert Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo") is None

    def test_get_issue_clears_state_when_closed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 4})
        gh = self._make_issue_gh(state="CLOSED")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert load_state(fido_dir) == {}

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
        save_state(fido_dir, {"issue": 12})
        gh = self._make_issue_gh(state="OPEN")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "alice/proj")
        gh.view_issue.assert_called_once_with("alice/proj", 12)

    def test_get_issue_logs_info_when_closed(self, tmp_path: Path, caplog) -> None:
        import logging

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 9})
        gh = self._make_issue_gh(state="CLOSED")
        with caplog.at_level(logging.INFO, logger="kennel"):
            Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert "advancing" in caplog.text

    def test_get_issue_state_preserved_when_open(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 5})
        gh = self._make_issue_gh(state="OPEN")
        Worker(tmp_path, gh).get_current_issue(fido_dir, "owner/repo")
        assert load_state(fido_dir) == {"issue": 5}

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
            patch.object(worker, "find_or_create_pr", return_value=(1, "my-branch")),
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
            patch.object(worker, "find_or_create_pr", return_value=(1, "my-branch")),
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
            patch.object(worker, "find_or_create_pr", return_value=(1, "my-branch")),
        ):
            worker.run()
        gh.view_issue.assert_called_once_with("owner/repo", 3)

    def test_run_calls_find_or_create_pr_when_issue_found(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "My task", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_focp = MagicMock(return_value=(42, "my-branch"))
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=8),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", mock_focp),
        ):
            worker.run()
        mock_focp.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 8, "My task")

    def test_run_returns_0_when_find_or_create_pr_returns_none(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Done", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=2),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=None),
        ):
            assert worker.run() == 0


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
        assert load_state(fido_dir) == {"issue": 7}

    def test_does_not_save_state_when_no_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_issues.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "set_status"):
            worker.find_next_issue(fido_dir, self._make_repo_ctx())
        assert load_state(fido_dir) == {}

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
        script = create_compact_script(fido_dir)
        assert script.exists()

    def test_returns_path_in_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script == fido_dir / "compact.sh"

    def test_script_is_executable(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script.stat().st_mode & 0o111  # any execute bit

    def test_script_has_shebang(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert script.read_text().startswith("#!/usr/bin/env bash\n")

    def test_script_references_sub_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        from kennel.worker import _sub_dir

        assert str(_sub_dir()) in script.read_text()

    def test_script_contains_post_compact_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
        assert "PostCompact" in script.read_text()

    def test_script_contains_md_glob(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        script = create_compact_script(fido_dir)
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
        assert load_state(fido_dir) == {}

    def test_returns_state_when_present(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 42}))
        assert load_state(fido_dir) == {"issue": 42}

    def test_returns_dict_with_arbitrary_keys(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "state.json").write_text(json.dumps({"issue": 7, "extra": "val"}))
        result = load_state(fido_dir)
        assert result["issue"] == 7
        assert result["extra"] == "val"


class TestSaveState:
    def test_creates_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 5})
        assert (fido_dir / "state.json").exists()

    def test_roundtrips_with_load_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 99})
        assert load_state(fido_dir) == {"issue": 99}

    def test_overwrites_existing_state(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 1})
        save_state(fido_dir, {"issue": 2})
        assert load_state(fido_dir) == {"issue": 2}


class TestClearState:
    def test_removes_state_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 3})
        clear_state(fido_dir)
        assert not (fido_dir / "state.json").exists()

    def test_noop_when_absent(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # Should not raise
        clear_state(fido_dir)

    def test_load_returns_empty_after_clear(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        save_state(fido_dir, {"issue": 10})
        clear_state(fido_dir)
        assert load_state(fido_dir) == {}


class TestBuildPrompt:
    """Tests for build_prompt."""

    def _setup_sub_dir(self, tmp_path: Path) -> Path:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "persona.md").write_text("I am Fido, a very good dog.")
        (sub / "task.md").write_text("Implement the task carefully.")
        return sub

    def test_returns_system_and_prompt_paths(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, prompt_file = build_prompt(fido_dir, "task", "context")
        assert sys_file == fido_dir / "system"
        assert prompt_file == fido_dir / "prompt"

    def test_system_file_contains_persona(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "context")
        assert "I am Fido" in sys_file.read_text()

    def test_system_file_contains_skill(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "context")
        assert "Implement the task carefully." in sys_file.read_text()

    def test_system_file_joins_with_blank_line(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "context")
        content = sys_file.read_text()
        assert "I am Fido, a very good dog.\n\nImplement the task carefully." in content

    def test_prompt_file_contains_context(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            _, prompt_file = build_prompt(fido_dir, "task", "do the work")
        assert "do the work" in prompt_file.read_text()

    def test_system_file_ends_with_newline(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "ctx")
        assert sys_file.read_text().endswith("\n")

    def test_prompt_file_ends_with_newline(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        with patch("kennel.worker._sub_dir", return_value=sub):
            _, prompt_file = build_prompt(fido_dir, "task", "ctx")
        assert prompt_file.read_text().endswith("\n")

    def test_uses_correct_subskill_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        (sub / "ci.md").write_text("Fix the CI failure.")
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "ci", "ctx")
        assert "Fix the CI failure." in sys_file.read_text()
        assert "Implement the task carefully." not in sys_file.read_text()

    def test_strips_trailing_whitespace_from_persona(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        sub = self._setup_sub_dir(tmp_path)
        (sub / "persona.md").write_text("Persona text\n\n\n")
        with patch("kennel.worker._sub_dir", return_value=sub):
            sys_file, _ = build_prompt(fido_dir, "task", "ctx")
        content = sys_file.read_text()
        assert content.startswith("Persona text\n\n")
        assert not content.startswith("Persona text\n\n\n\n")


class TestClaudeStart:
    """Tests for claude_start."""

    def _setup_fido_dir(self, tmp_path: Path) -> Path:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "system").write_text("system prompt")
        (fido_dir / "prompt").write_text("user prompt")
        return fido_dir

    def test_returns_session_id_on_success(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        output = '{"type":"result","session_id":"sess-abc"}'
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=output),
            patch(
                "kennel.claude.extract_session_id",
                return_value="sess-abc",
            ),
        ):
            result = claude_start(fido_dir)
        assert result == "sess-abc"

    def test_returns_empty_when_extract_fails(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=""),
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            result = claude_start(fido_dir)
        assert result == ""

    def test_calls_print_prompt_from_file_with_correct_files(
        self, tmp_path: Path
    ) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_start(fido_dir)
        mock_ppf.assert_called_once_with(
            fido_dir / "system",
            fido_dir / "prompt",
            "claude-sonnet-4-6",
            300,
        )

    def test_passes_custom_model(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_start(fido_dir, model="claude-opus-4-6")
        assert mock_ppf.call_args[0][2] == "claude-opus-4-6"

    def test_passes_custom_timeout(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_start(fido_dir, timeout=600)
        assert mock_ppf.call_args[0][3] == 600

    def test_default_model_is_sonnet(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_start(fido_dir)
        assert mock_ppf.call_args[0][2] == "claude-sonnet-4-6"

    def test_default_timeout_is_300(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_start(fido_dir)
        assert mock_ppf.call_args[0][3] == 300

    def test_passes_output_to_extract_session_id(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        raw = '{"type":"result","session_id":"xyz"}'
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=raw),
            patch("kennel.claude.extract_session_id", return_value="xyz") as mock_ext,
        ):
            claude_start(fido_dir)
        mock_ext.assert_called_once_with(raw)


class TestClaudeRun:
    """Tests for claude_run."""

    def _setup_fido_dir(self, tmp_path: Path) -> Path:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (fido_dir / "system").write_text("system")
        (fido_dir / "prompt").write_text("prompt")
        return fido_dir

    # ── Resume path ────────────────────────────────────────────────────────

    def test_resume_returns_existing_session_id(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with patch("kennel.worker.claude.resume_session", return_value="output text"):
            session_id, _ = claude_run(fido_dir, session_id="existing-id")
        assert session_id == "existing-id"

    def test_resume_returns_output(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with patch("kennel.worker.claude.resume_session", return_value="stream output"):
            _, output = claude_run(fido_dir, session_id="sid")
        assert output == "stream output"

    def test_resume_calls_resume_session_with_correct_args(
        self, tmp_path: Path
    ) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with patch("kennel.worker.claude.resume_session", return_value="") as mock_rs:
            claude_run(
                fido_dir,
                session_id="my-session",
                model="claude-opus-4-6",
                timeout=120,
            )
        mock_rs.assert_called_once_with(
            "my-session", fido_dir / "prompt", "claude-opus-4-6", 120
        )

    def test_resume_does_not_call_print_prompt_from_file(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch("kennel.worker.claude.resume_session", return_value=""),
            patch("kennel.worker.claude.print_prompt_from_file") as mock_ppf,
        ):
            claude_run(fido_dir, session_id="sid")
        mock_ppf.assert_not_called()

    # ── Start path ─────────────────────────────────────────────────────────

    def test_start_returns_new_session_id(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        raw = '{"type":"result","session_id":"new-sess"}'
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=raw),
            patch(
                "kennel.claude.extract_session_id",
                return_value="new-sess",
            ),
        ):
            session_id, _ = claude_run(fido_dir)
        assert session_id == "new-sess"

    def test_start_returns_raw_output(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        raw = '{"type":"result","session_id":"s"}'
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=raw),
            patch("kennel.claude.extract_session_id", return_value="s"),
        ):
            _, output = claude_run(fido_dir)
        assert output == raw

    def test_start_calls_print_prompt_from_file(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_run(fido_dir)
        mock_ppf.assert_called_once_with(
            fido_dir / "system",
            fido_dir / "prompt",
            "claude-sonnet-4-6",
            300,
        )

    def test_start_does_not_call_resume_session(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=""),
            patch("kennel.claude.extract_session_id", return_value=""),
            patch("kennel.worker.claude.resume_session") as mock_rs,
        ):
            claude_run(fido_dir)
        mock_rs.assert_not_called()

    def test_start_returns_empty_session_id_on_failure(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_from_file", return_value=""),
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            session_id, _ = claude_run(fido_dir)
        assert session_id == ""

    def test_default_model_is_sonnet(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_run(fido_dir)
        assert mock_ppf.call_args[0][2] == "claude-sonnet-4-6"

    def test_default_timeout_is_300(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_from_file", return_value=""
            ) as mock_ppf,
            patch("kennel.claude.extract_session_id", return_value=""),
        ):
            claude_run(fido_dir)
        assert mock_ppf.call_args[0][3] == 300

    def test_passes_custom_model_to_resume(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with patch("kennel.worker.claude.resume_session", return_value="") as mock_rs:
            claude_run(fido_dir, session_id="sid", model="claude-haiku-4-5-20251001")
        assert mock_rs.call_args[0][2] == "claude-haiku-4-5-20251001"

    def test_passes_custom_timeout_to_resume(self, tmp_path: Path) -> None:
        fido_dir = self._setup_fido_dir(tmp_path)
        with patch("kennel.worker.claude.resume_session", return_value="") as mock_rs:
            claude_run(fido_dir, session_id="sid", timeout=90)
        assert mock_rs.call_args[0][3] == 90


class TestSanitizeSlug:
    """Tests for _sanitize_slug."""

    def test_lowercases_input(self) -> None:
        assert _sanitize_slug("AddFeature", "fallback text") == "addfeature"

    def test_replaces_special_chars_with_hyphens(self) -> None:
        result = _sanitize_slug("add feature test", "fallback")
        assert result == "add-feature-test"

    def test_strips_leading_trailing_hyphens(self) -> None:
        result = _sanitize_slug("  add-feature  ", "fallback")
        assert result == "add-feature"

    def test_truncates_at_40_chars(self) -> None:
        long_input = "a" * 50
        result = _sanitize_slug(long_input, "fallback")
        assert len(result) <= 40

    def test_falls_back_when_too_short(self) -> None:
        result = _sanitize_slug("ab", "fix the login bug")
        assert "fix" in result or "login" in result or "bug" in result

    def test_falls_back_strips_closes_clause(self) -> None:
        result = _sanitize_slug("x", "fix bug (closes #42)")
        assert "42" not in result
        assert "closes" not in result

    def test_three_char_slug_is_valid(self) -> None:
        result = _sanitize_slug("abc", "fallback text")
        assert result == "abc"

    def test_two_char_slug_falls_back(self) -> None:
        result = _sanitize_slug("ab", "fix login issue")
        assert len(result) >= 3

    def test_empty_raw_falls_back(self) -> None:
        result = _sanitize_slug("", "implement auth")
        assert len(result) >= 3

    def test_fallback_also_sanitized(self) -> None:
        result = _sanitize_slug("x", "Add New Feature!")
        assert result == result.lower()
        assert "!" not in result


class TestGit:
    """Tests for Worker._git helper."""

    def test_calls_subprocess_run_with_git_prefix(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            worker._git(["status"])
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "git"
        assert args[1] == "status"

    def test_passes_work_dir_as_cwd(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            worker._git(["status"])
        assert mock_run.call_args[1]["cwd"] == tmp_path

    def test_check_true_by_default(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            worker._git(["status"])
        assert mock_run.call_args[1]["check"] is True

    def test_check_false_propagated(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            worker._git(["status"], check=False)
        assert mock_run.call_args[1]["check"] is False

    def test_propagates_called_process_error(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            with pytest.raises(subprocess.CalledProcessError):
                worker._git(["checkout", "no-such-branch"])

    def test_capture_output_and_text_set(self, tmp_path: Path) -> None:
        worker = Worker(tmp_path, MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            worker._git(["log"])
        assert mock_run.call_args[1]["capture_output"] is True
        assert mock_run.call_args[1]["text"] is True


class TestBuildPrBody:
    """Tests for Worker._build_pr_body."""

    def _make_worker(self, tmp_path: Path) -> "Worker":
        return Worker(tmp_path, MagicMock())

    def _pending_task(self, title: str) -> dict:
        return {"id": "1", "title": title, "status": "pending"}

    def test_returns_string(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="PR desc."),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("Fix the thing (closes #1)", 1)
        assert isinstance(result, str)

    def test_contains_work_queue_start_marker(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "<!-- WORK_QUEUE_START -->" in result

    def test_contains_work_queue_end_marker(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "<!-- WORK_QUEUE_END -->" in result

    def test_pending_tasks_shown_as_checkboxes(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        pending = [self._pending_task("Write tests"), self._pending_task("Fix lint")]
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=pending),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "- [ ] Write tests" in result
        assert "- [ ] Fix lint" in result

    def test_first_task_has_next_marker(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        pending = [self._pending_task("First task"), self._pending_task("Second task")]
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=pending),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "- [ ] First task **→ next**" in result

    def test_second_task_has_no_next_marker(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        pending = [self._pending_task("First task"), self._pending_task("Second task")]
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=pending),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "- [ ] Second task **→ next**" not in result
        assert "- [ ] Second task\n" in result or result.endswith("- [ ] Second task")

    def test_no_tasks_shows_placeholder(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "<!-- no tasks yet -->" in result

    def test_falls_back_to_plain_when_claude_returns_empty(
        self, tmp_path: Path
    ) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value=""),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("Fix auth", 7)
        assert "Working on: Fix auth" in result

    def test_contains_separator(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "---" in result

    def test_calls_claude_print_prompt_with_opus(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="d") as mock_pp,
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            worker._build_pr_body("req", 1)
        assert mock_pp.call_args[1]["model"] == "claude-opus-4-6"

    def test_system_prompt_includes_issue_number(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="d") as mock_pp,
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            worker._build_pr_body("req", 99)
        assert "99" in mock_pp.call_args[1]["system_prompt"]

    def test_prompt_includes_persona(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt", return_value="d") as mock_pp,
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido, a very good dog.")
            worker._build_pr_body("req", 1)
        assert "I am Fido, a very good dog." in mock_pp.call_args[1]["prompt"]

    def test_skips_completed_tasks(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        task_list = [
            {"id": "1", "title": "Done task", "status": "completed"},
            {"id": "2", "title": "Pending task", "status": "pending"},
        ]
        with (
            patch("kennel.worker.claude.print_prompt", return_value="d"),
            patch("kennel.worker.tasks.list_tasks", return_value=task_list),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "Done task" not in result
        assert "Pending task" in result

    def test_falls_back_gracefully_when_persona_missing(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        missing = tmp_path / "nosuchdir"
        with (
            patch("kennel.worker.claude.print_prompt", return_value="d") as mock_pp,
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=missing),
        ):
            worker._build_pr_body("req", 1)
        # Should not raise; persona becomes empty string
        assert mock_pp.called


class TestFindOrCreatePr:
    """Tests for Worker.find_or_create_pr."""

    def _make_worker(self, tmp_path: Path) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _make_repo_ctx(
        self,
        repo: str = "owner/proj",
        owner: str = "owner",
        repo_name: str = "proj",
        gh_user: str = "fido-bot",
        default_branch: str = "main",
    ) -> "RepoContext":
        from kennel.worker import RepoContext

        return RepoContext(
            repo=repo,
            owner=owner,
            repo_name=repo_name,
            gh_user=gh_user,
            default_branch=default_branch,
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "fido"
        d.mkdir()
        return d

    def _open_pr(self, number: int = 10, slug: str = "fix-bug") -> dict:
        return {"number": number, "headRefName": slug, "state": "OPEN"}

    def _merged_pr(self, number: int = 10, slug: str = "fix-bug") -> dict:
        return {"number": number, "headRefName": slug, "state": "MERGED"}

    def _closed_pr(self, number: int = 10, slug: str = "fix-bug") -> dict:
        return {"number": number, "headRefName": slug, "state": "CLOSED"}

    # --- Merged PR path ---

    def test_merged_pr_returns_none(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr()
        fido_dir = self._fido_dir(tmp_path)
        result = worker.find_or_create_pr(
            fido_dir, self._make_repo_ctx(), 5, "Fix the thing"
        )
        assert result is None

    def test_merged_pr_closes_issue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr(number=77)
        fido_dir = self._fido_dir(tmp_path)
        worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        gh.close_issue.assert_called_once_with("owner/proj", 5)

    def test_merged_pr_clears_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr()
        fido_dir = self._fido_dir(tmp_path)
        save_state(fido_dir, {"issue": 5})
        worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert load_state(fido_dir) == {}

    def test_merged_pr_logs_info(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr(number=33)
        fido_dir = self._fido_dir(tmp_path)
        with caplog.at_level(logging.INFO, logger="kennel"):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "merged" in caplog.text

    # --- Open PR (resume) path ---

    def test_open_pr_returns_pr_number_and_slug(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-branch")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.tasks.list_tasks", return_value=["a task"]),
        ):
            result = worker.find_or_create_pr(
                fido_dir, self._make_repo_ctx(), 5, "title"
            )
        assert result == (20, "my-branch")

    def test_open_pr_logs_resuming(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="fix-stuff")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.tasks.list_tasks", return_value=["a task"]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "resuming" in caplog.text

    def test_open_pr_runs_setup_when_no_tasks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        mock_start = MagicMock(return_value="sess-1")
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.claude_start", mock_start),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_called_once_with(fido_dir, "setup", ANY)
        mock_start.assert_called_once_with(fido_dir)

    def test_open_pr_skips_setup_when_tasks_exist(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        with (
            patch.object(worker, "_git"),
            patch(
                "kennel.worker.tasks.list_tasks",
                return_value=[{"title": "t", "status": "pending"}],
            ),
            patch("kennel.worker.build_prompt", mock_build),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_not_called()

    def test_open_pr_checkout_fallback_on_error(self, tmp_path: Path) -> None:
        """If git checkout slug fails, try checkout -b --track."""
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(slug="br")
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(args)
            if args == ["checkout", "br"]:
                raise subprocess.CalledProcessError(1, "git")
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.tasks.list_tasks", return_value=["t"]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert ["checkout", "-b", "br", "--track", "origin/br"] in git_calls

    def test_open_pr_fetches_before_checkout(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(slug="br")
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(args)
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.tasks.list_tasks", return_value=["t"]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert git_calls[0] == ["fetch", "origin"]

    # --- No PR (new branch) path ---

    def test_no_pr_returns_pr_number_and_slug(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/55"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="fix-bug"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value="sess"),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            result = worker.find_or_create_pr(
                fido_dir, self._make_repo_ctx(), 5, "Fix the bug"
            )
        assert result is not None
        pr_number, slug = result
        assert pr_number == 55
        assert slug == "fix-bug"

    def test_no_pr_logs_new_branch(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="do-work"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "new branch" in caplog.text

    def test_no_pr_calls_setup(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        mock_start = MagicMock(return_value="s")
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="do-work"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.claude_start", mock_start),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_called_once_with(fido_dir, "setup", ANY)
        mock_start.assert_called_once_with(fido_dir)

    def test_no_pr_creates_pr_with_correct_params(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/99"
        fido_dir = self._fido_dir(tmp_path)
        repo_ctx = self._make_repo_ctx(repo="owner/proj", default_branch="main")
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="do-work"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="pr-body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            worker.find_or_create_pr(fido_dir, repo_ctx, 7, "Do the work")
        gh.create_pr.assert_called_once_with(
            "owner/proj",
            "Do the work (closes #7)",
            "pr-body",
            "main",
            "do-work",
        )

    def test_no_pr_git_operations_in_order(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(list(args))
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.claude.generate_branch_name", return_value="do-work"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert git_calls[0] == ["fetch", "origin"]
        assert git_calls[-2] == ["commit", "--allow-empty", "-m", "wip: start"]
        assert git_calls[-1] == ["push", "-u", "origin", "do-work"]

    def test_no_pr_checkout_fallback_when_branch_exists(self, tmp_path: Path) -> None:
        """checkout -b fails (branch exists) → fall back to checkout."""
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(list(args))
            if args[0:2] == ["checkout", "-b"]:
                raise subprocess.CalledProcessError(128, "git")
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch("kennel.worker.claude.generate_branch_name", return_value="slug"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert ["checkout", "slug"] in git_calls

    def test_no_pr_slug_sanitized(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        git_calls = []

        def side_effect(args, check=True):  # noqa: ARG001
            git_calls.append(list(args))
            return MagicMock()

        with (
            patch.object(worker, "_git", side_effect=side_effect),
            patch(
                "kennel.worker.claude.generate_branch_name",
                return_value="Add New Feature!",
            ),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result is not None
        _, slug = result
        assert slug == slug.lower()
        assert "!" not in slug

    # --- Closed PR (fall through) path ---

    def test_closed_pr_creates_fresh_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._closed_pr(number=5, slug="old-br")
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/6"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="new-br"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            result = worker.find_or_create_pr(
                fido_dir, self._make_repo_ctx(), 5, "title"
            )
        assert result is not None
        pr_number, _ = result
        assert pr_number == 6

    def test_closed_pr_logs_message(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._closed_pr(number=77)
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/78"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="slug"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "closed" in caplog.text

    def test_no_pr_logs_pr_number(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/42"
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="work"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value=""),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert "42" in caplog.text


class TestSeedTasksFromPrBody:
    """Tests for Worker.seed_tasks_from_pr_body."""

    def _make_worker(self, tmp_path: Path) -> tuple["Worker", MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _pr_with_queue(self, *task_titles: str) -> dict:
        lines = "\n".join(f"- [ ] {t}" for t in task_titles)
        body = (
            f"Description.\n\n"
            f"<!-- WORK_QUEUE_START -->\n{lines}\n<!-- WORK_QUEUE_END -->"
        )
        return {"body": body}

    def test_noop_when_tasks_exist(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        with patch("kennel.worker.tasks.list_tasks", return_value=[{"title": "t"}]):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        gh.get_pr.assert_not_called()

    def test_fetches_pr_body_when_no_tasks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": ""}
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.seed_tasks_from_pr_body("owner/repo", 42)
        gh.get_pr.assert_called_once_with("owner/repo", 42)

    def test_noop_when_no_markers_in_body(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": "No markers here."}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_noop_when_no_unchecked_tasks_in_queue(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "<!-- no tasks yet -->\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_adds_single_task_from_body(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue("Fix the bug")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_called_once_with(tmp_path, "Fix the bug")

    def test_adds_multiple_tasks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue(
            "Task one", "Task two", "Task three"
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert mock_add.call_count == 3

    def test_strips_next_marker(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": (
                "<!-- WORK_QUEUE_START -->\n"
                "- [ ] First task **→ next**\n"
                "- [ ] Second task\n"
                "<!-- WORK_QUEUE_END -->"
            )
        }
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        titles = [call.args[1] for call in mock_add.call_args_list]
        assert titles[0] == "First task"
        assert "**→ next**" not in titles[0]

    def test_correct_task_titles_in_order(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue("Write the tests", "Fix the lint")
        received: list[str] = []
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch(
                "kennel.worker.tasks.add_task",
                side_effect=lambda wd, t: received.append(t),
            ),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 5)
        assert received == ["Write the tests", "Fix the lint"]

    def test_noop_when_body_is_none(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"body": None}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task") as mock_add,
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        mock_add.assert_not_called()

    def test_logs_info_with_task_count(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_with_queue("T1", "T2")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker.tasks.add_task"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert "seeded" in caplog.text
        assert "2" in caplog.text

    def test_does_not_log_when_no_tasks_found(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "body": "<!-- WORK_QUEUE_START -->\n<!-- WORK_QUEUE_END -->"
        }
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.seed_tasks_from_pr_body("owner/repo", 1)
        assert "seeded" not in caplog.text


class TestRunSeedTasksIntegration:
    """Tests for seed_tasks_from_pr_body being called from Worker.run()."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

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

    def test_seed_called_after_find_or_create_pr(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "My task", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_seed = MagicMock()
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=(42, "fix-bug")),
            patch.object(worker, "seed_tasks_from_pr_body", mock_seed),
        ):
            worker.run()
        mock_seed.assert_called_once_with("owner/repo", 42)

    def test_seed_not_called_when_find_or_create_pr_returns_none(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Done", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_seed = MagicMock()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=3),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=None),
            patch.object(worker, "seed_tasks_from_pr_body", mock_seed),
        ):
            worker.run()
        mock_seed.assert_not_called()


class TestExtractRunId:
    """Tests for Worker._extract_run_id."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def test_extracts_id_from_standard_url(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._extract_run_id(
                "https://github.com/owner/repo/actions/runs/123456789/job/987"
            )
            == "123456789"
        )

    def test_extracts_id_from_url_without_job(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert (
            w._extract_run_id("https://github.com/owner/repo/actions/runs/42") == "42"
        )

    def test_returns_empty_when_no_run_id(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._extract_run_id("https://github.com/owner/repo/actions") == ""

    def test_returns_empty_for_empty_link(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._extract_run_id("") == ""


class TestFilterCiThreads:
    """Tests for Worker._filter_ci_threads."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _make_threads_data(self, nodes: list) -> dict:
        return {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }

    def _make_node(
        self,
        *,
        resolved: bool = False,
        first_author: str = "reviewer",
        first_body: str = "CI is failing",
        last_author: str = "reviewer",
        last_body: str = "CI is failing",
        url: str = "https://example.com/comment",
    ) -> dict:
        return {
            "isResolved": resolved,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": first_author},
                        "body": first_body,
                        "url": url,
                    },
                    {
                        "author": {"login": last_author},
                        "body": last_body,
                        "url": url,
                    },
                ]
            },
        }

    def test_returns_empty_when_no_threads(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._filter_ci_threads({}, "fido-bot", "test") == []

    def test_excludes_resolved_threads(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(resolved=True, first_body="ci failing")
        data = self._make_threads_data([node])
        assert w._filter_ci_threads(data, "reviewer", "test") == []

    def test_excludes_threads_where_last_author_is_gh_user(
        self, tmp_path: Path
    ) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(last_author="fido-bot", last_body="ci fix")
        data = self._make_threads_data([node])
        assert w._filter_ci_threads(data, "fido-bot", "test") == []

    def test_excludes_threads_without_ci_keywords(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_body="style nit", last_body="please rename var")
        data = self._make_threads_data([node])
        assert w._filter_ci_threads(data, "fido-bot", "mycheck") == []

    def test_includes_thread_matching_check_name(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_body="mycheck is red", last_body="please fix")
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "mycheck")
        assert len(result) == 1

    def test_includes_thread_mentioning_ci(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_body="CI broke after your commit")
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "unrelated-check")
        assert len(result) == 1

    def test_includes_thread_mentioning_lint(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_body="lint errors in this file")
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "check")
        assert len(result) == 1

    def test_includes_thread_mentioning_format(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_body="format issue here")
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "check")
        assert len(result) == 1

    def test_maps_fields_correctly(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(
            first_author="alice",
            first_body="CI is red",
            last_author="bob",
            last_body="still red",
            url="https://github.com/x",
        )
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "ci")
        assert result[0] == {
            "first_author": "alice",
            "first_body": "CI is red",
            "last_author": "bob",
            "last_body": "still red",
            "url": "https://github.com/x",
        }

    def test_case_insensitive_keyword_match(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_body="LINT errors found")
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "check")
        assert len(result) == 1

    def test_single_comment_node(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = {
            "isResolved": False,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": "alice"},
                        "body": "ci failed",
                        "url": "https://example.com",
                    }
                ]
            },
        }
        data = self._make_threads_data([node])
        result = w._filter_ci_threads(data, "fido-bot", "check")
        assert len(result) == 1
        assert result[0]["first_author"] == "alice"
        assert result[0]["last_author"] == "alice"

    def test_skips_nodes_with_empty_comments(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = {"isResolved": False, "comments": {"nodes": []}}
        data = self._make_threads_data([node])
        assert w._filter_ci_threads(data, "fido-bot", "check") == []


class TestHandleCi:
    """Tests for Worker.handle_ci."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self) -> RepoContext:
        return RepoContext(
            repo="owner/repo",
            owner="owner",
            repo_name="repo",
            gh_user="fido-bot",
            default_branch="main",
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sub").mkdir(exist_ok=True)
        return d

    def test_returns_false_when_no_checks(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = []
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False

    def test_returns_false_when_all_checks_pass(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "SUCCESS", "link": ""},
        ]
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch") is False

    def test_returns_true_on_failure(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": "runs/99"},
        ]
        gh.get_run_log.return_value = "line1\nline2"
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            result = worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_returns_true_on_error_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "lint", "state": "ERROR", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            result = worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_calls_set_status_with_check_name_and_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "unit-tests", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 7, "branch")
        mock_status.assert_called_once_with("Fixing CI: unit-tests on PR #7")

    def test_fetches_run_log_when_run_id_present(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {
                "name": "build",
                "state": "FAILURE",
                "link": "https://github.com/owner/repo/actions/runs/55555/job/1",
            }
        ]
        gh.get_run_log.return_value = "some log output"
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        gh.get_run_log.assert_called_once_with("owner/repo", "55555")

    def test_skips_run_log_fetch_when_no_run_id(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "build", "state": "FAILURE", "link": "no-run-id-here"},
        ]
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        gh.get_run_log.assert_not_called()

    def test_truncates_log_to_last_200_lines(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": "runs/1"},
        ]
        long_log = "\n".join(f"line {i}" for i in range(300))
        gh.get_run_log.return_value = long_log
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        captured_context = {}
        with (
            patch.object(worker, "set_status"),
            patch(
                "kennel.worker.build_prompt",
                side_effect=lambda fd, sk, ctx: captured_context.update({"ctx": ctx}),
            ),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        log_section = captured_context["ctx"].split("Failure log")[1]
        assert "line 99\n" not in log_section  # line 99 is before the last 200
        assert "line 100\n" in log_section  # line 100 starts the last 200

    def test_fetches_review_threads_for_repo(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 42, "branch")
        gh.get_review_threads.assert_called_once_with("owner", "repo", 42)

    def test_builds_ci_prompt(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "my-check", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 5, "fix-branch")
        mock_bp.assert_called_once()
        _, subskill, context = mock_bp.call_args[0]
        assert subskill == "ci"
        assert "my-check" in context
        assert "fix-branch" in context
        assert "PR: 5" in context

    def test_runs_claude(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sess-1", "")) as mock_cr,
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_called_once_with(fido_dir)

    def test_completes_ci_task_by_title(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "my-check", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_complete.assert_called_once_with(tmp_path, "CI failure: my-check")

    def test_spawns_sync_script(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "test", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen") as mock_popen,
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_popen.assert_called_once()
        popen_args = mock_popen.call_args[0][0]
        assert popen_args[0] == "bash"
        assert "sync-tasks.sh" in popen_args[1]

    def test_picks_first_failing_check(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "pass-check", "state": "SUCCESS", "link": ""},
            {"name": "fail-check", "state": "FAILURE", "link": ""},
            {"name": "err-check", "state": "ERROR", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        # First failing check is "fail-check"
        mock_status.assert_called_once_with("Fixing CI: fail-check on PR #1")

    def test_logs_ci_failing(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.pr_checks.return_value = [
            {"name": "flaky", "state": "FAILURE", "link": ""},
        ]
        gh.get_run_log.return_value = ""
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("subprocess.Popen"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        assert "flaky" in caplog.text


class TestRunHandleCiIntegration:
    """Tests that Worker.run() calls handle_ci and respects its return value."""

    def _make_gh(self) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.get_default_branch.return_value = "main"
        gh.get_pr.return_value = {"body": ""}
        return gh

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

    def test_handle_ci_called_with_pr_and_slug(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_handle_ci = MagicMock(return_value=False)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=(42, "fix-bug")),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", mock_handle_ci),
        ):
            worker.run()
        mock_handle_ci.assert_called_once_with(
            mock_ctx.fido_dir, repo_ctx, 42, "fix-bug"
        )

    def test_returns_1_when_ci_handled(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=(42, "fix-bug")),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=True),
        ):
            result = worker.run()
        assert result == 1

    def test_returns_0_when_ci_passes(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        repo_ctx = self._make_mock_repo_ctx()
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(worker, "discover_repo_context", return_value=repo_ctx),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=7),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=(42, "fix-bug")),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", return_value=False),
        ):
            result = worker.run()
        assert result == 0

    def test_handle_ci_not_called_when_find_or_create_pr_returns_none(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Done", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_handle_ci = MagicMock(return_value=False)
        with (
            patch.object(worker, "create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch.object(worker, "setup_hooks", return_value=("c", "s")),
            patch.object(worker, "teardown_hooks"),
            patch.object(worker, "get_current_issue", return_value=3),
            patch.object(worker, "post_pickup_comment"),
            patch.object(worker, "find_or_create_pr", return_value=None),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch.object(worker, "handle_ci", mock_handle_ci),
        ):
            worker.run()
        mock_handle_ci.assert_not_called()
