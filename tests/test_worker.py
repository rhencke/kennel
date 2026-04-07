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
    acquire_lock,
    clear_state,
    create_compact_script,
    create_context,
    load_state,
    resolve_git_dir,
    run,
    save_state,
    setup_hooks,
    teardown_hooks,
)


class TestResolveGitDir:
    def test_returns_path(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/some/repo/.git\n")
            result = resolve_git_dir(tmp_path)
        assert result == Path("/some/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  /a/b/.git  \n")
            result = resolve_git_dir(tmp_path)
        assert result == Path("/a/b/.git")

    def test_calls_correct_command(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/a/.git")
            resolve_git_dir(tmp_path)
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
                resolve_git_dir(tmp_path)


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
        with patch("kennel.worker.resolve_git_dir", return_value=git_dir):
            ctx = create_context(tmp_path)
        assert isinstance(ctx, WorkerContext)
        assert ctx.work_dir == tmp_path
        assert ctx.git_dir == git_dir
        assert ctx.fido_dir == git_dir / "fido"
        ctx.lock_fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        with patch("kennel.worker.resolve_git_dir", return_value=git_dir):
            ctx = create_context(tmp_path)
        assert ctx.fido_dir.is_dir()
        ctx.lock_fd.close()

    def test_propagates_lock_held(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        fido_dir = git_dir / "fido"
        with patch("kennel.worker.resolve_git_dir", return_value=git_dir):
            fd1 = acquire_lock(fido_dir)
            try:
                with pytest.raises(LockHeld):
                    create_context(tmp_path)
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
        with patch("kennel.worker.create_context", side_effect=LockHeld("held")):
            assert Worker(tmp_path, gh).run() == 2

    def test_run_returns_0_on_success(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks"),
        ):
            assert worker.run() == 0

    def test_run_logs_warning_on_lock_held(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        with patch("kennel.worker.create_context", side_effect=LockHeld("held")):
            with caplog.at_level(logging.WARNING, logger="kennel"):
                Worker(tmp_path, gh).run()
        assert "another fido" in caplog.text

    def test_run_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks"),
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
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks"),
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
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks", mock_teardown),
        ):
            worker.run()
        mock_teardown.assert_called_once()

    def test_run_setup_hooks_called_with_fido_dir(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_setup = MagicMock(return_value=("c", "s"))
        gh = self._make_gh()
        worker = Worker(tmp_path, gh)
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch.object(
                worker, "discover_repo_context", return_value=self._make_mock_repo_ctx()
            ),
            patch("kennel.worker.setup_hooks", mock_setup),
            patch("kennel.worker.teardown_hooks"),
        ):
            worker.run()
        mock_setup.assert_called_once_with(tmp_path, mock_ctx.fido_dir)


class TestRun:
    """Tests for the module-level run() convenience wrapper."""

    def test_creates_worker_with_github_and_delegates(self, tmp_path: Path) -> None:
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
        compact_cmd, sync_cmd = setup_hooks(tmp_path, fido_dir)
        assert compact_cmd.startswith("bash ")
        assert sync_cmd.startswith("bash ")

    def test_compact_cmd_references_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, _ = setup_hooks(tmp_path, fido_dir)
        assert "compact.sh" in compact_cmd

    def test_sync_cmd_references_sync_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        _, sync_cmd = setup_hooks(tmp_path, fido_dir)
        assert "sync-tasks.sh" in sync_cmd

    def test_sync_cmd_includes_work_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        _, sync_cmd = setup_hooks(tmp_path, fido_dir)
        assert str(tmp_path) in sync_cmd

    def test_creates_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        setup_hooks(tmp_path, fido_dir)
        assert (fido_dir / "compact.sh").exists()

    def test_adds_hooks_to_settings(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        setup_hooks(tmp_path, fido_dir)
        settings = tmp_path / ".claude" / "settings.local.json"
        assert settings.exists()
        cfg = json.loads(settings.read_text())
        assert "hooks" in cfg

    def test_gitexcludes_settings(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        setup_hooks(tmp_path, fido_dir)
        exclude = tmp_path / ".git" / "info" / "exclude"
        assert ".claude/settings.local.json" in exclude.read_text()


class TestTeardownHooks:
    def test_removes_compact_script(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, sync_cmd = setup_hooks(tmp_path, fido_dir)
        teardown_hooks(tmp_path, fido_dir, compact_cmd, sync_cmd)
        assert not (fido_dir / "compact.sh").exists()

    def test_removes_hooks_from_settings(self, tmp_path: Path) -> None:
        import json

        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        (tmp_path / ".git" / "info").mkdir(parents=True)
        compact_cmd, sync_cmd = setup_hooks(tmp_path, fido_dir)
        teardown_hooks(tmp_path, fido_dir, compact_cmd, sync_cmd)
        settings = tmp_path / ".claude" / "settings.local.json"
        cfg = json.loads(settings.read_text())
        assert "hooks" not in cfg

    def test_noop_when_compact_script_missing(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # Should not raise even when compact.sh does not exist
        teardown_hooks(tmp_path, fido_dir, "bash /x/compact.sh", "bash sync.sh &")

    def test_noop_when_settings_missing(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fido_dir.mkdir()
        # No settings file created — should not raise
        teardown_hooks(tmp_path, fido_dir, "bash /x/compact.sh", "bash sync.sh &")


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
