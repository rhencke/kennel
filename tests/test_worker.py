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
    WorkerThread,
    _apply_queue_to_body,
    _auto_complete_ask_tasks,
    _format_work_queue,
    _pick_next_task,
    _resolve_git_dir,
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
    should_rerequest_review,
    sync_tasks,
    sync_tasks_background,
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
                "kennel.worker.claude.generate_status_with_session",
                return_value=("writing tests", "sess-1"),
            ),
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("writing tests")
        gh.set_user_status.assert_called_once_with("writing tests", "🐕", busy=True)

    def test_set_status_busy_false_forwarded(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=("napping", "sess-1"),
            ),
            patch("kennel.worker.claude.generate_status_emoji", return_value="😴"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("napping", busy=False)
        gh.set_user_status.assert_called_once_with("napping", "😴", busy=False)

    def test_set_status_skips_when_claude_returns_empty(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=("", ""),
            ),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("idle")
        gh.set_user_status.assert_not_called()

    def test_set_status_emoji_fallback_when_empty(self, tmp_path: Path) -> None:
        # generate_status_emoji returns empty → :dog: fallback
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=("Sniffing out endpoints", "sess-1"),
            ),
            patch("kennel.worker.claude.generate_status_emoji", return_value=""),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("idle")
        gh.set_user_status.assert_called_once_with(
            "Sniffing out endpoints", ":dog:", busy=True
        )

    def test_set_status_text_truncated_to_80_chars(self, tmp_path: Path) -> None:
        # All retries fail (return empty) → fall back to truncation
        gh = self._make_gh()
        long_text = "x" * 100
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=(long_text, "sess-1"),
            ),
            patch("kennel.worker.claude.resume_status", return_value=""),
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("something")
        called_text = gh.set_user_status.call_args[0][0]
        assert len(called_text) == 80

    def test_set_status_retries_when_text_exceeds_80_chars(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        long_text = "y" * 90
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=(long_text, "sess-99"),
            ),
            patch(
                "kennel.worker.claude.resume_status",
                return_value="shorter text",
            ) as mock_resume,
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("something")
        mock_resume.assert_called_once_with("sess-99", ANY)
        gh.set_user_status.assert_called_once_with("shorter text", "🐕", busy=True)

    def test_set_status_stops_retrying_when_text_fits(self, tmp_path: Path) -> None:
        # Second retry produces short text → no third retry
        gh = self._make_gh()
        long_text = "z" * 90
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=(long_text, "sess-7"),
            ),
            patch(
                "kennel.worker.claude.resume_status",
                side_effect=["z" * 85, "good"],
            ) as mock_resume,
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("something")
        assert mock_resume.call_count == 2
        gh.set_user_status.assert_called_once_with("good", "🐕", busy=True)

    def test_set_status_retries_up_to_3_times_max(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        long_text = "w" * 90
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=(long_text, "sess-3"),
            ),
            patch(
                "kennel.worker.claude.resume_status",
                return_value=long_text,
            ) as mock_resume,
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("something")
        assert mock_resume.call_count == 3

    def test_set_status_skips_retry_when_no_session_id(self, tmp_path: Path) -> None:
        # No session_id → retry loop is skipped, truncation applied directly
        gh = self._make_gh()
        long_text = "v" * 100
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=(long_text, ""),
            ),
            patch("kennel.worker.claude.resume_status") as mock_resume,
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            Worker(tmp_path, gh).set_status("something")
        mock_resume.assert_not_called()
        called_text = gh.set_user_status.call_args[0][0]
        assert len(called_text) == 80

    def test_set_status_logs_warning_on_empty_response(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=("", ""),
            ),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.WARNING, logger="kennel"):
                Worker(tmp_path, gh).set_status("idle")
        assert "empty" in caplog.text

    def test_set_status_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=("fetching", "sess-1"),
            ),
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
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
                "kennel.worker.claude.generate_status_with_session",
                return_value=("working", "sess-1"),
            ) as mock_gen,
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=missing_dir),
        ):
            Worker(tmp_path, gh).set_status("working")
        # persona file missing — generate_status_with_session still called with empty persona
        prompt_arg = mock_gen.call_args[1]["prompt"]
        assert "What you're doing right now: working" in prompt_arg

    def test_set_status_passes_system_prompt_to_generate_status_with_session(
        self, tmp_path: Path
    ) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status_with_session",
                return_value=("working", "sess-1"),
            ) as mock_gen,
            patch("kennel.worker.claude.generate_status_emoji", return_value="🐕"),
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
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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
        assert "sync-tasks" in sync_cmd

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
        assert "kennel sync-tasks" in sync_cmd

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
            patch("kennel.worker.claude.print_prompt_json", return_value="PR desc."),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("Fix the thing (closes #1)", 1)
        assert isinstance(result, str)

    def test_contains_work_queue_start_marker(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "<!-- WORK_QUEUE_START -->" in result

    def test_contains_work_queue_end_marker(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
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
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
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
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
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
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=pending),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "- [ ] Second task **→ next**" not in result
        assert "- [ ] Second task\n" in result or result.endswith("- [ ] Second task")

    def test_next_marker_follows_pick_next_task_priority(self, tmp_path: Path) -> None:
        """CI failure task second in list should still get the → next marker."""
        worker = self._make_worker(tmp_path)
        regular = self._pending_task("Regular work")
        ci = {"id": "2", "title": "CI failure: lint", "status": "pending"}
        with (
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[regular, ci]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "- [ ] CI failure: lint **→ next**" in result
        assert "- [ ] Regular work **→ next**" not in result

    def test_no_tasks_shows_placeholder(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
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
            patch("kennel.worker.claude.print_prompt_json", return_value=""),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("Fix auth", 7)
        assert "Working on: Fix auth" in result

    def test_contains_separator(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch("kennel.worker.claude.print_prompt_json", return_value="desc"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            result = worker._build_pr_body("req", 1)
        assert "---" in result

    def test_calls_claude_print_prompt_json_with_opus(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_json", return_value="d"
            ) as mock_pp,
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            worker._build_pr_body("req", 1)
        assert mock_pp.call_args[1]["model"] == "claude-opus-4-6"

    def test_system_prompt_includes_issue_number(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_json", return_value="d"
            ) as mock_pp,
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("")
            worker._build_pr_body("req", 99)
        assert "99" in mock_pp.call_args[1]["system_prompt"]

    def test_prompt_includes_persona(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with (
            patch(
                "kennel.worker.claude.print_prompt_json", return_value="d"
            ) as mock_pp,
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
            patch(
                "kennel.worker.claude.print_prompt_json", return_value="d"
            ) as mock_pp,
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
        with patch.object(worker, "_git"):
            result = worker.find_or_create_pr(
                fido_dir, self._make_repo_ctx(), 5, "Fix the thing"
            )
        assert result is None

    def test_merged_pr_clears_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr()
        fido_dir = self._fido_dir(tmp_path)
        save_state(fido_dir, {"issue": 5})
        with patch.object(worker, "_git"):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        assert load_state(fido_dir) == {}

    def test_merged_pr_deletes_remote_branch(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr(slug="fix-bug")
        fido_dir = self._fido_dir(tmp_path)
        with patch.object(worker, "_git") as mock_git:
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_git.assert_called_once_with(
            ["push", "origin", "--delete", "fix-bug"], check=False
        )

    def test_merged_pr_logs_info(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._merged_pr(number=33)
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
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
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.claude_start", mock_start),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        mock_build.assert_called_once_with(fido_dir, "setup", ANY)
        mock_start.assert_called_once_with(fido_dir)

    def test_open_pr_setup_context_includes_work_dir(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.claude_start", return_value="sess"),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        _, _, context = mock_build.call_args.args
        assert f"Work dir: {tmp_path}" in context

    def test_open_pr_setup_no_tasks_returns_none(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "seed_tasks_from_pr_body"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value="sess"),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result is None

    def test_open_pr_seeds_from_pr_body_before_setup(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        # seed_tasks_from_pr_body populates tasks → setup not called
        call_order = []
        with (
            patch.object(worker, "_git"),
            patch(
                "kennel.worker.tasks.list_tasks",
                side_effect=[[], [{"id": "1", "title": "t", "status": "pending"}]],
            ),
            patch.object(
                worker,
                "seed_tasks_from_pr_body",
                side_effect=lambda *a: call_order.append("seed"),
            ),
            patch(
                "kennel.worker.build_prompt",
                side_effect=lambda *a: call_order.append("setup"),
            ),
            patch("kennel.worker.claude_start", return_value="sess"),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result == (20, "my-br")
        assert "seed" in call_order
        assert "setup" not in call_order

    def test_open_pr_setup_produces_tasks_returns_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = self._open_pr(number=20, slug="my-br")
        fido_dir = self._fido_dir(tmp_path)
        task = {"title": "t", "status": "pending"}
        call_count = 0

        def list_tasks_side_effect(_work_dir):
            nonlocal call_count
            call_count += 1
            return [] if call_count == 1 else [task]

        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.tasks.list_tasks", side_effect=list_tasks_side_effect),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value="sess"),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result == (20, "my-br")

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
            patch(
                "kennel.worker.tasks.list_tasks",
                return_value=[{"title": "Do thing", "status": "pending"}],
            ),
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

    def test_no_pr_setup_context_includes_work_dir(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        gh.create_pr.return_value = "https://github.com/owner/proj/pull/1"
        fido_dir = self._fido_dir(tmp_path)
        mock_build = MagicMock()
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="do-work"),
            patch("kennel.worker.build_prompt", mock_build),
            patch("kennel.worker.claude_start", return_value="s"),
            patch.object(worker, "_build_pr_body", return_value="body"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "title")
        _, _, context = mock_build.call_args.args
        assert f"Work dir: {tmp_path}" in context

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
            patch(
                "kennel.worker.tasks.list_tasks",
                return_value=[{"title": "t", "status": "pending"}],
            ),
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
            patch(
                "kennel.worker.tasks.list_tasks",
                return_value=[{"title": "t", "status": "pending"}],
            ),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result is not None
        _, slug = result
        assert slug == slug.lower()
        assert "!" not in slug

    def test_no_pr_setup_no_tasks_returns_none(self, tmp_path: Path) -> None:
        """New-PR path: setup produces no tasks → return None, skip PR creation."""
        worker, gh = self._make_worker(tmp_path)
        gh.find_pr.return_value = None
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch.object(worker, "_git"),
            patch("kennel.worker.claude.generate_branch_name", return_value="fix-bug"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_start", return_value="sess"),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
        ):
            result = worker.find_or_create_pr(fido_dir, self._make_repo_ctx(), 5, "t")
        assert result is None
        gh.create_pr.assert_not_called()

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
            patch(
                "kennel.worker.tasks.list_tasks",
                return_value=[{"title": "t", "status": "pending"}],
            ),
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
            patch(
                "kennel.worker.tasks.list_tasks",
                return_value=[{"title": "t", "status": "pending"}],
            ),
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
            patch.object(worker, "handle_ci", return_value=False),
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks") as mock_sync,
        ):
            worker.handle_ci(fido_dir, self._repo_ctx(), 1, "branch")
        mock_sync.assert_called_once_with(tmp_path, gh)

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
            patch("kennel.worker.sync_tasks"),
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
            patch("kennel.worker.sync_tasks"),
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
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


class TestFilterThreads:
    """Tests for Worker._filter_threads."""

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
        node_id: str = "thread-1",
        first_author: str = "owner",
        first_db_id: int = 1,
        first_body: str = "please fix this",
        last_author: str | None = None,
        last_body: str = "please fix this",
        url: str = "https://example.com/comment",
        extra_comments: list | None = None,
    ) -> dict:
        """Build a review-thread node.

        If *last_author* is given (and differs from first_author or last_body
        differs from first_body) a second comment is appended.  *extra_comments*
        are inserted between first and last.
        """
        first = {
            "author": {"login": first_author},
            "body": first_body,
            "url": url,
            "databaseId": first_db_id,
        }
        comments: list = [first]
        if extra_comments:
            comments.extend(extra_comments)
        if last_author is not None and (
            last_author != first_author or last_body != first_body
        ):
            comments.append(
                {
                    "author": {"login": last_author},
                    "body": last_body,
                    "url": url,
                    "databaseId": first_db_id + len(comments),
                }
            )
        return {
            "id": node_id,
            "isResolved": resolved,
            "comments": {"nodes": comments},
        }

    def test_returns_empty_when_no_nodes(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        assert w._filter_threads({}, "fido-bot", "owner") == []

    def test_excludes_resolved_threads(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(resolved=True)
        data = self._make_threads_data([node])
        assert w._filter_threads(data, "fido-bot", "owner") == []

    def test_excludes_threads_where_last_author_is_gh_user(
        self, tmp_path: Path
    ) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(last_author="fido-bot", last_body="done")
        data = self._make_threads_data([node])
        assert w._filter_threads(data, "fido-bot", "owner") == []

    def test_excludes_threads_where_last_author_is_neither_owner_nor_bot(
        self, tmp_path: Path
    ) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(last_author="random-user", last_body="comment")
        data = self._make_threads_data([node])
        assert w._filter_threads(data, "fido-bot", "owner") == []

    def test_includes_thread_from_owner(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(last_author="owner")
        data = self._make_threads_data([node])
        result = w._filter_threads(data, "fido-bot", "owner")
        assert len(result) == 1

    def test_includes_thread_from_bot(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(last_author="my-app[bot]", last_body="bot comment")
        data = self._make_threads_data([node])
        result = w._filter_threads(data, "fido-bot", "owner")
        assert len(result) == 1

    def test_maps_fields_correctly(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(
            node_id="tid-42",
            first_author="owner",
            first_db_id=99,
            first_body="first comment",
            last_author="owner",
            last_body="last comment",
            url="https://github.com/x",
        )
        data = self._make_threads_data([node])
        result = w._filter_threads(data, "fido-bot", "owner")
        assert result[0]["id"] == "tid-42"
        assert result[0]["first_author"] == "owner"
        assert result[0]["first_db_id"] == 99
        assert result[0]["first_body"] == "first comment"
        assert result[0]["last_author"] == "owner"
        assert result[0]["last_body"] == "last comment"
        assert result[0]["url"] == "https://github.com/x"
        assert result[0]["total"] == 2

    def test_is_bot_true_when_first_author_ends_with_bot(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_author="my-app[bot]", last_author="owner")
        data = self._make_threads_data([node])
        result = w._filter_threads(data, "fido-bot", "owner")
        assert result[0]["is_bot"] is True

    def test_is_bot_false_when_first_author_is_human(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(first_author="owner", last_author="owner")
        data = self._make_threads_data([node])
        result = w._filter_threads(data, "fido-bot", "owner")
        assert result[0]["is_bot"] is False

    def test_excludes_threads_with_no_comments(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = {"id": "x", "isResolved": False, "comments": {"nodes": []}}
        data = self._make_threads_data([node])
        assert w._filter_threads(data, "fido-bot", "owner") == []

    def test_total_counts_all_comments(self, tmp_path: Path) -> None:
        w = self._make_worker(tmp_path)
        node = self._make_node(
            first_author="owner",
            last_author="owner",
            last_body="final comment",
            extra_comments=[
                {
                    "author": {"login": "owner"},
                    "body": "mid",
                    "url": "u",
                    "databaseId": 2,
                }
            ],
        )
        data = self._make_threads_data([node])
        result = w._filter_threads(data, "fido-bot", "owner")
        assert result[0]["total"] == 3


class TestResolveAddressedThreads:
    """Tests for Worker.resolve_addressed_threads."""

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

    def _make_threads_data(self, nodes: list) -> dict:
        return {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }

    def _make_node(
        self,
        *,
        resolved: bool = False,
        node_id: str = "thread-1",
        last_author: str = "fido-bot",
    ) -> dict:
        return {
            "id": node_id,
            "isResolved": resolved,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": "owner"},
                        "body": "please fix",
                        "url": "https://example.com",
                        "databaseId": 1,
                    },
                    {
                        "author": {"login": last_author},
                        "body": "done",
                        "url": "https://example.com",
                        "databaseId": 2,
                    },
                ]
            },
        }

    def test_returns_false_when_no_threads(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = {}
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_returns_false_when_all_threads_resolved(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._make_node(resolved=True)
        gh.get_review_threads.return_value = self._make_threads_data([node])
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_returns_false_when_last_author_is_not_gh_user(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._make_node(last_author="owner")
        gh.get_review_threads.return_value = self._make_threads_data([node])
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_returns_false_when_thread_has_no_comments(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = {"id": "t1", "isResolved": False, "comments": {"nodes": []}}
        gh.get_review_threads.return_value = self._make_threads_data([node])
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is False
        gh.resolve_thread.assert_not_called()

    def test_resolves_thread_where_gh_user_is_last_author(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._make_node(node_id="tid-99", last_author="fido-bot")
        gh.get_review_threads.return_value = self._make_threads_data([node])
        result = worker.resolve_addressed_threads(self._repo_ctx(), 1)
        assert result is True
        gh.resolve_thread.assert_called_once_with("tid-99")

    def test_resolves_multiple_threads(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        nodes = [
            self._make_node(node_id="t1", last_author="fido-bot"),
            self._make_node(node_id="t2", last_author="fido-bot"),
        ]
        gh.get_review_threads.return_value = self._make_threads_data(nodes)
        result = worker.resolve_addressed_threads(self._repo_ctx(), 5)
        assert result is True
        assert gh.resolve_thread.call_count == 2
        gh.resolve_thread.assert_any_call("t1")
        gh.resolve_thread.assert_any_call("t2")

    def test_skips_already_resolved_among_mixed(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        nodes = [
            self._make_node(node_id="t1", last_author="fido-bot", resolved=True),
            self._make_node(node_id="t2", last_author="fido-bot"),
        ]
        gh.get_review_threads.return_value = self._make_threads_data(nodes)
        worker.resolve_addressed_threads(self._repo_ctx(), 1)
        gh.resolve_thread.assert_called_once_with("t2")

    def test_calls_get_review_threads_with_correct_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = {}
        worker.resolve_addressed_threads(self._repo_ctx(), 42)
        gh.get_review_threads.assert_called_once_with("owner", "repo", 42)


class TestHandleReviewFeedback:
    """Tests for Worker.handle_review_feedback."""

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
        return d

    def _pr_data(
        self,
        *,
        state: str = "CHANGES_REQUESTED",
        submitted_at: str = "2024-01-01T12:00:00Z",
        review_body: str = "Please fix the typo.",
        commit_date: str = "2024-01-01T10:00:00Z",
    ) -> dict:
        return {
            "reviews": [
                {
                    "author": {"login": "owner"},
                    "state": state,
                    "submittedAt": submitted_at,
                    "body": review_body,
                }
            ],
            "commits": [
                {
                    "messageHeadline": "Fix bug",
                    "oid": "abc",
                    "committedDate": commit_date,
                }
            ],
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
            "body": "",
        }

    def test_returns_false_when_no_reviews(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {"reviews": [], "commits": [], "body": ""}
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_false_when_no_owner_reviews(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = {
            "reviews": [
                {
                    "author": {"login": "other-user"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T12:00:00Z",
                    "body": "change this",
                }
            ],
            "commits": [],
            "body": "",
        }
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_false_when_last_review_not_changes_requested(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data(state="APPROVED")
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_false_when_commits_newer_than_review(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data(
            submitted_at="2024-01-01T10:00:00Z",
            commit_date="2024-01-01T12:00:00Z",
        )
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_false_when_review_body_empty(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data(review_body="")
        fido_dir = self._fido_dir(tmp_path)
        assert (
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
            is False
        )

    def test_returns_true_when_changes_requested_no_newer_commits(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data()
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            result = worker.handle_review_feedback(
                fido_dir, self._repo_ctx(), 1, "branch"
            )
        assert result is True

    def test_builds_task_prompt(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data(review_body="fix the nit")
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 5, "my-branch")
        mock_bp.assert_called_once()
        _, subskill, context = mock_bp.call_args[0]
        assert subskill == "task"
        assert "PR: 5" in context
        assert "my-branch" in context

    def test_runs_claude(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data()
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sess-1", "")) as mock_cr,
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_called_once_with(fido_dir)

    def test_completes_task_by_title(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data()
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
            patch("kennel.worker.sync_tasks"),
        ):
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
        mock_complete.assert_called_once_with(
            tmp_path, "Address review feedback from owner"
        )

    def test_spawns_sync_script(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data()
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks") as mock_sync,
        ):
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_logs_review_feedback(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        gh.get_pr.return_value = self._pr_data()
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_review_feedback(fido_dir, self._repo_ctx(), 1, "branch")
        assert "review feedback" in caplog.text


class TestHandleThreads:
    """Tests for Worker.handle_threads."""

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
        return d

    def _threads_data_with_nodes(self, nodes: list) -> dict:
        return {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }

    def _open_thread_node(
        self, *, last_author: str = "owner", first_author: str = "owner"
    ) -> dict:
        return {
            "id": "thread-1",
            "isResolved": False,
            "comments": {
                "nodes": [
                    {
                        "author": {"login": first_author},
                        "body": "please fix",
                        "url": "https://example.com",
                        "databaseId": 1,
                    },
                    {
                        "author": {"login": last_author},
                        "body": "still open",
                        "url": "https://example.com",
                        "databaseId": 2,
                    },
                ]
            },
        }

    def test_returns_false_when_no_threads(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        assert worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch") is False

    def test_returns_true_when_threads_exist(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = self._threads_data_with_nodes([node])
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch("kennel.worker.sync_tasks_background"),
        ):
            result = worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_calls_get_review_threads_with_correct_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        gh.get_review_threads.return_value = {}
        fido_dir = self._fido_dir(tmp_path)
        worker.handle_threads(fido_dir, self._repo_ctx(), 42, "branch")
        gh.get_review_threads.assert_called_once_with("owner", "repo", 42)

    def test_builds_comments_prompt(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = self._threads_data_with_nodes([node])
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch("kennel.worker.sync_tasks_background"),
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 5, "my-branch")
        mock_bp.assert_called_once()
        _, subskill, context = mock_bp.call_args[0]
        assert subskill == "comments"
        assert "PR: 5" in context
        assert "my-branch" in context

    def test_runs_claude(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = self._threads_data_with_nodes([node])
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sess-1", "")) as mock_cr,
            patch("kennel.worker.sync_tasks_background"),
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        mock_cr.assert_called_once_with(fido_dir)

    def test_spawns_sync_script(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = self._threads_data_with_nodes([node])
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.sync_tasks_background") as mock_sync,
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_logs_thread_count(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        node = self._open_thread_node()
        gh.get_review_threads.return_value = self._threads_data_with_nodes([node])
        fido_dir = self._fido_dir(tmp_path)
        with (
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch("kennel.worker.sync_tasks_background"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_threads(fido_dir, self._repo_ctx(), 1, "branch")
        assert "unresolved threads" in caplog.text


class TestRunReviewFeedbackIntegration:
    """Tests that Worker.run() calls handle_review_feedback after handle_ci."""

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

    def test_handle_review_feedback_called_when_ci_passes(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_rf = MagicMock(return_value=False)
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
            patch.object(worker, "handle_review_feedback", mock_rf),
            patch.object(worker, "handle_threads", return_value=False),
        ):
            worker.run()
        mock_rf.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug")

    def test_returns_1_when_review_feedback_handled(self, tmp_path: Path) -> None:
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
            patch.object(worker, "handle_review_feedback", return_value=True),
        ):
            result = worker.run()
        assert result == 1

    def test_handle_threads_not_called_when_review_feedback_handled(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_threads = MagicMock(return_value=False)
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
            patch.object(worker, "handle_review_feedback", return_value=True),
            patch.object(worker, "handle_threads", mock_threads),
        ):
            worker.run()
        mock_threads.assert_not_called()


class TestRunThreadsIntegration:
    """Tests that Worker.run() calls handle_threads after handle_review_feedback."""

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

    def test_handle_threads_called_when_review_feedback_passes(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_threads = MagicMock(return_value=False)
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", mock_threads),
        ):
            worker.run()
        mock_threads.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug")

    def test_returns_1_when_threads_handled(self, tmp_path: Path) -> None:
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=True),
        ):
            result = worker.run()
        assert result == 1

    def test_returns_0_when_no_work(self, tmp_path: Path) -> None:
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
        ):
            result = worker.run()
        assert result == 0


class TestPickNextTask:
    """Tests for the module-level _pick_next_task helper."""

    def _task(
        self,
        title: str,
        status: str = "pending",
        thread: dict | None = None,
    ) -> dict:
        t: dict = {"id": "x", "title": title, "status": status}
        if thread is not None:
            t["thread"] = thread
        return t

    def test_returns_none_for_empty_list(self) -> None:
        assert _pick_next_task([]) is None

    def test_returns_none_when_all_completed(self) -> None:
        tasks = [self._task("Fix it", status="completed")]
        assert _pick_next_task(tasks) is None

    def test_skips_ask_prefix_lowercase(self) -> None:
        tasks = [self._task("ask: clarify this")]
        assert _pick_next_task(tasks) is None

    def test_skips_ask_prefix_uppercase(self) -> None:
        tasks = [self._task("ASK: clarify this")]
        assert _pick_next_task(tasks) is None

    def test_skips_ask_prefix_mixed_case(self) -> None:
        tasks = [self._task("Ask: clarify this")]
        assert _pick_next_task(tasks) is None

    def test_skips_defer_prefix_lowercase(self) -> None:
        tasks = [self._task("defer: until later")]
        assert _pick_next_task(tasks) is None

    def test_skips_defer_prefix_uppercase(self) -> None:
        tasks = [self._task("DEFER: until later")]
        assert _pick_next_task(tasks) is None

    def test_returns_none_when_only_ask_and_defer(self) -> None:
        tasks = [
            self._task("ask: need info"),
            self._task("defer: not now"),
        ]
        assert _pick_next_task(tasks) is None

    def test_returns_regular_task(self) -> None:
        t = self._task("Implement feature X")
        assert _pick_next_task([t]) is t

    def test_ci_failure_takes_priority_over_regular(self) -> None:
        regular = self._task("Implement feature X")
        ci = self._task("CI failure: lint")
        assert _pick_next_task([regular, ci]) is ci

    def test_ci_failure_takes_priority_over_thread(self) -> None:
        thread_task = self._task("PR comment: fix nit", thread={"repo": "r", "pr": 1})
        ci = self._task("CI failure: tests")
        assert _pick_next_task([thread_task, ci]) is ci

    def test_thread_task_takes_priority_over_regular(self) -> None:
        regular = self._task("Regular work")
        thread_task = self._task(
            "PR comment: rename var", thread={"repo": "r", "pr": 1}
        )
        assert _pick_next_task([regular, thread_task]) is thread_task

    def test_returns_first_pending_when_no_special(self) -> None:
        first = self._task("First task")
        second = self._task("Second task")
        assert _pick_next_task([first, second]) is first

    def test_ignores_completed_when_selecting(self) -> None:
        completed = self._task("Done task", status="completed")
        pending = self._task("Pending task")
        assert _pick_next_task([completed, pending]) is pending

    def test_task_without_thread_key_not_thread_originated(self) -> None:
        """A task with no thread key is not treated as thread-originated."""
        no_thread = self._task("Regular task")
        assert no_thread.get("thread") is None
        result = _pick_next_task([no_thread])
        assert result is no_thread  # returned via fallthrough, not thread path


class TestEnsurePushed:
    """Tests for Worker.ensure_pushed."""

    def _make_worker(self, tmp_path: Path) -> Worker:
        return Worker(tmp_path, MagicMock())

    def _git_result(
        self, returncode: int = 0, stdout: str = "", stderr: str = ""
    ) -> MagicMock:
        r = MagicMock()
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def test_returns_none_when_already_in_sync(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        sha = "abc123"
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout=sha + "\n"),  # rev-parse HEAD
                self._git_result(stdout=sha + "\n"),  # rev-parse remote/slug
            ],
        ) as mock_git:
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is None
        assert mock_git.call_count == 2
        assert ["push", "-u", "origin", "my-branch"] not in [
            c[0][0] for c in mock_git.call_args_list
        ]

    def test_pushes_when_remote_ref_missing(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="abc\n"),  # rev-parse HEAD
                self._git_result(returncode=128),  # rev-parse remote/slug — not found
                self._git_result(),  # push succeeds
            ],
        ):
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is True

    def test_pushes_when_shas_differ(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="local-sha\n"),
                self._git_result(stdout="remote-sha\n"),
                self._git_result(),
            ],
        ):
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is True

    def test_returns_false_when_push_fails(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="local\n"),
                self._git_result(returncode=128),
                self._git_result(returncode=1, stderr="rejected"),
            ],
        ):
            result = worker.ensure_pushed("origin", "my-branch")
        assert result is False

    def test_logs_warning_on_push_failure(self, tmp_path: Path, caplog) -> None:
        import logging

        worker = self._make_worker(tmp_path)
        with (
            patch.object(
                worker,
                "_git",
                side_effect=[
                    self._git_result(stdout="local\n"),
                    self._git_result(returncode=128),
                    self._git_result(returncode=1, stderr="push rejected"),
                ],
            ),
            caplog.at_level(logging.WARNING, logger="kennel"),
        ):
            worker.ensure_pushed("origin", "br")
        assert "push failed" in caplog.text

    def test_passes_correct_remote_and_slug(self, tmp_path: Path) -> None:
        worker = self._make_worker(tmp_path)
        with patch.object(
            worker,
            "_git",
            side_effect=[
                self._git_result(stdout="sha1\n"),
                self._git_result(returncode=128),
                self._git_result(),
            ],
        ) as mock_git:
            worker.ensure_pushed("fork", "feature-branch")
        push_call = mock_git.call_args_list[2][0][0]
        assert push_call == ["push", "-u", "fork", "feature-branch"]


class TestExecuteTask:
    """Tests for Worker.execute_task."""

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
        return d

    def _pending_task(self, title: str) -> dict:
        return {"id": "t1", "title": title, "status": "pending"}

    @staticmethod
    def _git_with_new_commits():
        """Mock _git so rev-parse HEAD returns different SHAs before/after."""
        shas = iter(["aaa", "bbb"])
        orig = MagicMock()

        def side_effect(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = next(shas, "bbb") if args == ["rev-parse", "HEAD"] else ""
            result.stderr = ""
            return result

        orig.side_effect = side_effect
        return orig

    def test_returns_false_when_no_tasks(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is False

    def test_returns_true_when_task_found(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Implement feature")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sid", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "branch")
        assert result is True

    def test_calls_set_status_with_task_title(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Write the tests")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status") as mock_status,
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 5, "my-branch")
        mock_status.assert_called_once_with("Working on: Write the tests")

    def test_builds_task_prompt_with_correct_skill(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Fix the bug")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 7, "fix-branch")
        _, skill, _ = mock_bp.call_args[0]
        assert skill == "task"

    def test_context_includes_pr_and_repo(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Do work")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 42, "my-slug")
        _, _, context = mock_bp.call_args[0]
        assert "PR: 42" in context
        assert "Repo: owner/repo" in context
        assert "Branch: my-slug" in context
        assert "Upstream: origin/main" in context

    def test_context_includes_task_title(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("The special task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        _, _, context = mock_bp.call_args[0]
        assert "Task title: The special task" in context

    def test_context_includes_thread_metadata(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = {
            "id": "t1",
            "title": "Fix the thing",
            "status": "pending",
            "thread": {
                "repo": "owner/repo",
                "pr": 42,
                "comment_id": 12345,
                "url": "https://github.com/owner/repo/pull/42#discussion_r12345",
            },
        }
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 42, "br")
        _, _, context = mock_bp.call_args[0]
        assert "comment_id: 12345" in context
        assert "review comment" in context
        assert "Thread URL:" in context

    def test_context_omits_thread_when_no_thread(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Plain task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        _, _, context = mock_bp.call_args[0]
        assert "comment_id" not in context
        assert "review comment" not in context

    def test_calls_claude_run(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("sess", "")) as mock_run,
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_run.assert_called_once_with(fido_dir)

    def test_calls_ensure_pushed_with_origin_and_slug(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True) as mock_push,
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "my-slug")
        mock_push.assert_called_once_with("origin", "my-slug")

    def test_completes_task_by_title(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("My task title")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_called_once_with(tmp_path, "My task title")

    def test_skips_complete_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_not_called()

    def test_returns_true_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True

    def test_skips_sync_when_push_fails(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=False),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks") as mock_sync,
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_sync.assert_not_called()

    def test_completes_when_already_in_sync(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_complete.assert_called_once_with(tmp_path, "A task")

    def test_returns_true_when_already_in_sync(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            result = worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert result is True

    def test_syncs_when_already_in_sync(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=None),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks") as mock_sync,
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_syncs_work_queue_after_completion(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks") as mock_sync,
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        mock_sync.assert_called_once_with(tmp_path, gh)

    def test_logs_task_name(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Log me please")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert "Log me please" in caplog.text

    def test_logs_task_done_with_session_id(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch("kennel.worker.claude_run", return_value=("my-session", "")),
            patch.object(worker, "_git", self._git_with_new_commits()),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert "my-session" in caplog.text

    def test_resumes_session_until_commits_appear(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Big refactor")
        # head_before=aaa, head_after_1=aaa (no change), head_after_2=bbb (commit)
        shas = iter(["aaa", "aaa", "bbb"])
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout=next(shas, "bbb") if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch(
                "kennel.worker.claude_run",
                side_effect=[("sess-1", "output1"), ("sess-1", "output2")],
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        # First call: fresh start. Second call: resume with session_id.
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1][1]["session_id"] == "sess-1"

    def test_starts_fresh_when_no_session_id(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("A task")
        # No session_id on first run, retry starts fresh, second run produces commits
        shas = iter(["aaa", "aaa", "bbb"])
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout=next(shas, "bbb") if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt") as mock_bp,
            patch(
                "kennel.worker.claude_run",
                side_effect=[("", "output"), ("sess-2", "output2")],
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title"),
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        # build_prompt called twice: initial + fresh restart
        assert mock_bp.call_count == 2
        assert mock_run.call_count == 2

    def test_keeps_retrying_across_multiple_resumes(self, tmp_path: Path) -> None:
        worker, _ = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        task = self._pending_task("Big task")
        # 3 resumes before commits appear
        shas = iter(["aaa", "aaa", "aaa", "aaa", "bbb"])
        git_mock = MagicMock(
            side_effect=lambda args, **kw: MagicMock(
                returncode=0,
                stdout=next(shas, "bbb") if args == ["rev-parse", "HEAD"] else "",
                stderr="",
            )
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch.object(worker, "set_status"),
            patch("kennel.worker.build_prompt"),
            patch(
                "kennel.worker.claude_run",
                side_effect=[
                    ("sess-1", "o1"),
                    ("sess-1", "o2"),
                    ("sess-1", "o3"),
                    ("sess-1", "o4"),
                ],
            ) as mock_run,
            patch.object(worker, "_git", git_mock),
            patch.object(worker, "ensure_pushed", return_value=True),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
            patch("kennel.worker.sync_tasks"),
        ):
            worker.execute_task(fido_dir, self._repo_ctx(), 1, "br")
        assert mock_run.call_count == 4
        mock_complete.assert_called_once()


class TestRunExecuteTaskIntegration:
    """Tests that Worker.run() calls execute_task after handle_threads."""

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

    def test_execute_task_called_when_all_others_return_false(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_execute = MagicMock(return_value=False)
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", mock_execute),
        ):
            worker.run()
        mock_execute.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug")

    def test_returns_1_when_execute_task_done(self, tmp_path: Path) -> None:
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=True),
            patch.object(worker, "resolve_addressed_threads"),
        ):
            result = worker.run()
        assert result == 1

    def test_execute_task_not_called_when_threads_handled(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_execute = MagicMock(return_value=False)
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=True),
            patch.object(worker, "execute_task", mock_execute),
        ):
            worker.run()
        mock_execute.assert_not_called()


class TestShouldRerequestReview:
    """Tests for the should_rerequest_review module-level helper."""

    def _review(
        self,
        state: str = "CHANGES_REQUESTED",
        submitted_at: str = "",
    ) -> dict:
        r: dict = {"state": state}
        if submitted_at:
            r["submittedAt"] = submitted_at
        return r

    def _commit(self, committed_date: str) -> dict:
        return {"committedDate": committed_date}

    def test_empty_reviews_returns_false(self) -> None:
        assert should_rerequest_review([], []) is False

    def test_approved_returns_false(self) -> None:
        assert should_rerequest_review([self._review("APPROVED")], []) is False

    def test_commented_returns_false(self) -> None:
        assert should_rerequest_review([self._review("COMMENTED")], []) is False

    def test_changes_requested_no_dates_returns_true(self) -> None:
        assert should_rerequest_review([self._review()], []) is True

    def test_changes_requested_no_submitted_at_returns_true(self) -> None:
        commits = [self._commit("2024-01-02T12:00:00Z")]
        assert should_rerequest_review([self._review()], commits) is True

    def test_changes_requested_no_commits_returns_true(self) -> None:
        review = self._review(submitted_at="2024-01-02T12:00:00Z")
        assert should_rerequest_review([review], []) is True

    def test_review_older_than_commit_returns_true(self) -> None:
        """Review pre-dates latest commit — we addressed it."""
        review = self._review(submitted_at="2024-01-01T10:00:00Z")
        commits = [self._commit("2024-01-02T12:00:00Z")]
        assert should_rerequest_review([review], commits) is True

    def test_review_newer_than_commit_returns_false(self) -> None:
        """Review post-dates latest commit — new feedback, not yet addressed."""
        review = self._review(submitted_at="2024-01-02T12:00:00Z")
        commits = [self._commit("2024-01-01T10:00:00Z")]
        assert should_rerequest_review([review], commits) is False

    def test_uses_latest_commit_date(self) -> None:
        """Max commit date is used, not the first one."""
        review = self._review(submitted_at="2024-01-02T10:00:00Z")
        commits = [
            self._commit("2024-01-01T08:00:00Z"),
            self._commit("2024-01-03T08:00:00Z"),
        ]
        assert should_rerequest_review([review], commits) is True

    def test_uses_latest_owner_review(self) -> None:
        """Only the last review matters."""
        reviews = [
            self._review("APPROVED"),
            self._review("CHANGES_REQUESTED"),
        ]
        assert should_rerequest_review(reviews, []) is True

    def test_last_review_approved_overrides_earlier_changes_requested(self) -> None:
        reviews = [
            self._review("CHANGES_REQUESTED"),
            self._review("APPROVED"),
        ]
        assert should_rerequest_review(reviews, []) is False


class TestHandlePromoteMerge:
    """Tests for Worker.handle_promote_merge."""

    def _make_worker(self, tmp_path: Path) -> tuple[Worker, MagicMock]:
        gh = MagicMock()
        return Worker(tmp_path, gh), gh

    def _repo_ctx(self, owner: str = "rhencke") -> RepoContext:
        return RepoContext(
            repo=f"{owner}/myrepo",
            owner=owner,
            repo_name="myrepo",
            gh_user="fido-bot",
            default_branch="main",
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _reviews(
        self,
        owner: str = "rhencke",
        state: str = "APPROVED",
        is_draft: bool = False,
        commits: list | None = None,
    ) -> dict:
        return {
            "reviews": [{"author": {"login": owner}, "state": state}],
            "commits": commits if commits is not None else [],
            "isDraft": is_draft,
        }

    # --- get_reviews call ---

    def test_calls_get_reviews_with_correct_args(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 7, "my-branch", 3)
        gh.get_reviews.assert_called_once_with("rhencke/myrepo", 7)

    # --- merge on approval ---

    def test_approved_not_draft_no_pending_calls_get_pr(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.get_pr.assert_called_once_with("rhencke/myrepo", 9)

    def test_approved_not_draft_no_pending_merges_squash(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once_with("rhencke/myrepo", 9, squash=True)

    def test_approved_not_draft_no_pending_resets_tasks_json(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        tasks_file = fido_dir / "tasks.json"
        tasks_file.write_text('[{"id":"x","status":"completed"}]')
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert tasks_file.read_text() == "[]"

    def test_approved_not_draft_no_pending_clears_state(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        save_state(fido_dir, {"issue": 5})
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert load_state(fido_dir) == {}

    def test_approved_not_draft_no_pending_git_checkout_default(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["checkout", "main"] in calls

    def test_approved_not_draft_no_pending_git_pull_ff_only(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["pull", "origin", "main", "--ff-only"] in calls

    def test_approved_not_draft_no_pending_git_branch_delete(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["branch", "-d", "fix-slug"] in calls

    def test_approved_not_draft_no_pending_git_push_delete_remote_branch(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_git = MagicMock()
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git", mock_git),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix-slug", 5)
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert ["push", "origin", "--delete", "fix-slug"] in calls

    def test_approved_not_draft_no_pending_sets_merged_status(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        mock_status = MagicMock()
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status", mock_status),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        msg = mock_status.call_args[0][0]
        assert "Merged" in msg
        assert "9" in msg
        assert "5" in msg

    def test_approved_not_draft_no_pending_returns_1(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1

    def test_approved_not_draft_no_pending_blocked_enables_auto_merge(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_called_once_with("rhencke/myrepo", 9, squash=True, auto=True)

    def test_approved_not_draft_no_pending_blocked_returns_0(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_approved_with_pending_skips_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        pending = [{"id": "t1", "title": "Do work", "status": "pending"}]
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=pending),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    def test_approved_but_draft_skips_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=True)
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=completed),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    def test_not_approved_skips_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(state="COMMENTED", is_draft=False)
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    # --- changes requested ---

    def test_changes_requested_adds_reviewer(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_called_once_with("rhencke/myrepo", 9, "rhencke")

    def test_changes_requested_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_does_not_merge(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_merge.assert_not_called()

    def test_uses_latest_review_for_changes_requested(self, tmp_path: Path) -> None:
        """Later APPROVED overrides earlier CHANGES_REQUESTED."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {"author": {"login": "rhencke"}, "state": "CHANGES_REQUESTED"},
                {"author": {"login": "rhencke"}, "state": "APPROVED"},
            ],
            "commits": [],
            "isDraft": False,
        }
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        # APPROVED is latest — should merge, not re-request
        gh.pr_merge.assert_called_once()
        gh.add_pr_reviewer.assert_not_called()

    def test_changes_requested_newer_than_commit_skips_re_request(
        self, tmp_path: Path
    ) -> None:
        """Review submitted after latest commit — new feedback, don't re-request."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-01T10:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_not_called()

    def test_changes_requested_newer_than_commit_returns_0(
        self, tmp_path: Path
    ) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-01T10:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_older_than_commit_re_requests(
        self, tmp_path: Path
    ) -> None:
        """Review submitted before latest commit — we addressed it, re-request."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-02T12:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_called_once_with("rhencke/myrepo", 9, "rhencke")

    def test_changes_requested_older_than_commit_returns_0(
        self, tmp_path: Path
    ) -> None:
        """Regression for #124: after addressing feedback and pushing new commits,
        handle_promote_merge returns 0 (waiting for re-review, no more work)."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-02T12:00:00Z"}],
            "isDraft": False,
        }
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_changes_requested_no_submitted_at_re_requests(
        self, tmp_path: Path
    ) -> None:
        """No submittedAt on review — fall back to re-requesting."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_called_once()

    def test_changes_requested_no_commits_re_requests(self, tmp_path: Path) -> None:
        """No commits in data — fall back to re-requesting."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [],
            "isDraft": False,
        }
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_called_once()

    def test_changes_requested_newer_than_commit_logs_skip(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-02T12:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-01T10:00:00Z"}],
            "isDraft": False,
        }
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "skipping re-request" in caplog.text

    def test_changes_requested_owner_already_requested_skips_add(
        self, tmp_path: Path
    ) -> None:
        """Owner already in requested_reviewers — don't add again."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [
                {
                    "author": {"login": "rhencke"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2024-01-01T10:00:00Z",
                }
            ],
            "commits": [{"committedDate": "2024-01-02T12:00:00Z"}],
            "isDraft": False,
            "requestedReviewers": ["rhencke"],
        }
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_not_called()

    # --- draft promote ---

    def test_draft_no_completed_tasks_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    def test_draft_no_completed_tasks_does_not_promote(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_not_called()

    def test_draft_with_completed_tasks_calls_pr_ready(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.worker.tasks.list_tasks", return_value=completed):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.pr_ready.assert_called_once_with("rhencke/myrepo", 9)

    def test_draft_with_completed_tasks_adds_reviewer(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.worker.tasks.list_tasks", return_value=completed):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_called_once_with("rhencke/myrepo", 9, "rhencke")

    def test_draft_with_completed_tasks_returns_1(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.worker.tasks.list_tasks", return_value=completed):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1

    def test_draft_owner_already_requested_skips_add(self, tmp_path: Path) -> None:
        """Owner already in requested_reviewers — don't add again."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {
            "reviews": [],
            "commits": [],
            "isDraft": True,
            "requestedReviewers": ["rhencke"],
        }
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with patch("kennel.worker.tasks.list_tasks", return_value=completed):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        gh.add_pr_reviewer.assert_not_called()

    def test_draft_pending_tasks_ignored_for_promote_decision(
        self, tmp_path: Path
    ) -> None:
        """Pending tasks don't prevent promote — only completed count matters."""
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        tasks_list = [
            {"id": "t1", "title": "Done", "status": "completed"},
            {"id": "t2", "title": "Next", "status": "pending"},
        ]
        with patch("kennel.worker.tasks.list_tasks", return_value=tasks_list):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 1
        gh.pr_ready.assert_called_once()

    # --- idle / no work ---

    def test_not_draft_not_approved_idle_sets_status(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        mock_status = MagicMock()
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "set_status", mock_status),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        mock_status.assert_called_once_with("Napping — waiting for work", busy=False)

    def test_not_draft_not_approved_idle_returns_0(self, tmp_path: Path) -> None:
        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "set_status"),
        ):
            result = worker.handle_promote_merge(
                fido_dir, self._repo_ctx(), 9, "fix", 5
            )
        assert result == 0

    # --- logging ---

    def test_logs_review_status_check(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "review status" in caplog.text

    def test_logs_merge(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "CLEAN"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "_git"),
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "merging" in caplog.text

    def test_logs_auto_merge(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(is_draft=False)
        gh.get_pr.return_value = {"mergeStateStatus": "BLOCKED"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "auto-merge" in caplog.text

    def test_logs_changes_requested(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = self._reviews(
            state="CHANGES_REQUESTED", is_draft=False
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "changes requested" in caplog.text

    def test_logs_not_promoting(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "not promoting" in caplog.text

    def test_logs_marking_ready(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": True}
        completed = [{"id": "t1", "title": "Done", "status": "completed"}]
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=completed),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "marking ready" in caplog.text

    def test_logs_no_work(self, tmp_path: Path, caplog) -> None:
        import logging

        worker, gh = self._make_worker(tmp_path)
        fido_dir = self._fido_dir(tmp_path)
        gh.get_reviews.return_value = {"reviews": [], "commits": [], "isDraft": False}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch.object(worker, "set_status"),
            caplog.at_level(logging.INFO, logger="kennel"),
        ):
            worker.handle_promote_merge(fido_dir, self._repo_ctx(), 9, "fix", 5)
        assert "no work" in caplog.text


class TestRunPromoteMergeIntegration:
    """Tests that Worker.run() calls handle_promote_merge after execute_task."""

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

    def test_handle_promote_merge_called_when_execute_task_returns_false(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_hpm = MagicMock(return_value=0)
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", mock_hpm),
        ):
            worker.run()
        mock_hpm.assert_called_once_with(mock_ctx.fido_dir, repo_ctx, 42, "fix-bug", 7)

    def test_run_returns_0_when_handle_promote_merge_returns_0(
        self, tmp_path: Path
    ) -> None:
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=0),
        ):
            result = worker.run()
        assert result == 0

    def test_run_returns_1_when_handle_promote_merge_returns_1(
        self, tmp_path: Path
    ) -> None:
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=False),
            patch.object(worker, "handle_promote_merge", return_value=1),
        ):
            result = worker.run()
        assert result == 1

    def test_handle_promote_merge_not_called_when_execute_task_returns_true(
        self, tmp_path: Path
    ) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        gh = self._make_gh()
        gh.view_issue.return_value = {"title": "Fix it", "body": "", "state": "OPEN"}
        worker = Worker(tmp_path, gh)
        mock_hpm = MagicMock(return_value=0)
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
            patch.object(worker, "handle_review_feedback", return_value=False),
            patch.object(worker, "handle_threads", return_value=False),
            patch.object(worker, "execute_task", return_value=True),
            patch.object(worker, "resolve_addressed_threads"),
            patch.object(worker, "handle_promote_merge", mock_hpm),
        ):
            worker.run()
        mock_hpm.assert_not_called()


class TestResolveGitDirModuleLevel:
    def test_returns_path_from_stdout(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/some/repo/.git\n")
            result = _resolve_git_dir(tmp_path)
        assert result == Path("/some/repo/.git")

    def test_passes_absolute_git_dir_flag(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/repo/.git\n")
            _resolve_git_dir(tmp_path)
        args = mock_run.call_args[0][0]
        assert "--absolute-git-dir" in args

    def test_raises_on_subprocess_failure(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")
        ):
            with pytest.raises(subprocess.CalledProcessError):
                _resolve_git_dir(tmp_path)

    def test_passes_cwd(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/repo/.git\n")
            _resolve_git_dir(tmp_path)
        assert mock_run.call_args.kwargs["cwd"] == tmp_path


class TestFormatWorkQueue:
    def test_empty_list_returns_empty_string(self) -> None:
        assert _format_work_queue([]) == ""

    def test_pending_task_appears_as_unchecked(self) -> None:
        tasks = [{"title": "Do work", "status": "pending"}]
        result = _format_work_queue(tasks)
        assert "- [ ] Do work **→ next**" in result

    def test_first_pending_has_next_marker(self) -> None:
        tasks = [
            {"title": "Task A", "status": "pending"},
            {"title": "Task B", "status": "pending"},
        ]
        result = _format_work_queue(tasks)
        assert "Task A **→ next**" in result
        assert "Task B **→ next**" not in result

    def test_completed_tasks_appear_in_details(self) -> None:
        tasks = [{"title": "Done", "status": "completed"}]
        result = _format_work_queue(tasks)
        assert "<details>" in result
        assert "- [x] Done" in result

    def test_ci_failure_has_priority_over_others(self) -> None:
        tasks = [
            {"title": "Normal task", "status": "pending"},
            {"title": "CI failure: lint", "status": "pending"},
        ]
        result = _format_work_queue(tasks)
        lines = [ln for ln in result.splitlines() if "- [ ]" in ln]
        assert "CI failure: lint" in lines[0]
        assert "Normal task" in lines[1]

    def test_thread_task_has_priority_over_other_pending(self) -> None:
        tasks = [
            {"title": "Normal", "status": "pending"},
            {"title": "Thread task", "status": "pending", "thread": {"url": ""}},
        ]
        result = _format_work_queue(tasks)
        lines = [ln for ln in result.splitlines() if "- [ ]" in ln]
        assert "Thread task" in lines[0]
        assert "Normal" in lines[1]

    def test_thread_task_with_url_becomes_link(self) -> None:
        tasks = [
            {
                "title": "Fix it",
                "status": "pending",
                "thread": {"url": "https://github.com/comment/1"},
            }
        ]
        result = _format_work_queue(tasks)
        assert "[Fix it](https://github.com/comment/1)" in result

    def test_completed_count_in_summary(self) -> None:
        tasks = [
            {"title": "A", "status": "completed"},
            {"title": "B", "status": "completed"},
        ]
        result = _format_work_queue(tasks)
        assert "Completed (2)" in result

    def test_in_progress_treated_as_pending(self) -> None:
        tasks = [{"title": "Running", "status": "in_progress"}]
        result = _format_work_queue(tasks)
        assert "- [ ] Running" in result


class TestApplyQueueToBody:
    def test_replaces_queue_section(self) -> None:
        body = "Before\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->\nAfter"
        result = _apply_queue_to_body(body, "new")
        assert "new" in result
        assert "old" not in result

    def test_preserves_content_before_and_after(self) -> None:
        body = "Before\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->\nAfter"
        result = _apply_queue_to_body(body, "new")
        assert result.startswith("Before")
        assert result.endswith("After")

    def test_returns_body_unchanged_when_no_start_marker(self) -> None:
        body = "No markers here\n<!-- WORK_QUEUE_END -->"
        result = _apply_queue_to_body(body, "queue")
        assert result == body

    def test_returns_body_unchanged_when_no_end_marker(self) -> None:
        body = "<!-- WORK_QUEUE_START -->\nno end marker"
        result = _apply_queue_to_body(body, "queue")
        assert result == body

    def test_returns_body_unchanged_when_no_markers(self) -> None:
        body = "plain body text"
        result = _apply_queue_to_body(body, "queue")
        assert result == body


class TestAutoCompleteAskTasks:
    def _ask_task(self, comment_id: int) -> dict:
        return {
            "title": "ASK: some question",
            "status": "pending",
            "thread": {"comment_id": comment_id, "url": "https://example.com"},
        }

    def _resolved_node(self, db_id: int) -> dict:
        return {
            "isResolved": True,
            "comments": {"nodes": [{"databaseId": db_id}]},
        }

    def _threads_data(self, nodes: list) -> dict:
        return {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
        }

    def test_no_ask_tasks_does_nothing(self, tmp_path: Path) -> None:
        gh = MagicMock()
        with patch("kennel.worker.tasks.list_tasks", return_value=[]):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        gh.get_review_threads.assert_not_called()

    def test_completes_ask_task_when_thread_resolved(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = self._ask_task(42)
        gh.get_review_threads.return_value = self._threads_data(
            [self._resolved_node(42)]
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_called_once_with(tmp_path, "ASK: some question")

    def test_does_not_complete_ask_task_when_thread_not_resolved(
        self, tmp_path: Path
    ) -> None:
        gh = MagicMock()
        task = self._ask_task(42)
        gh.get_review_threads.return_value = self._threads_data(
            [{"isResolved": False, "comments": {"nodes": [{"databaseId": 42}]}}]
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()

    def test_get_review_threads_exception_logged(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = self._ask_task(1)
        gh.get_review_threads.side_effect = RuntimeError("api fail")
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()

    def test_non_ask_tasks_ignored(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = {
            "title": "Normal task",
            "status": "pending",
            "thread": {"comment_id": 1},
        }
        gh.get_review_threads.return_value = self._threads_data(
            [self._resolved_node(1)]
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()

    def test_ask_task_without_thread_ignored(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = {"title": "ASK: question", "status": "pending"}
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()
        gh.get_review_threads.assert_not_called()

    def test_resolved_node_without_database_id_skipped(self, tmp_path: Path) -> None:
        gh = MagicMock()
        task = self._ask_task(42)
        gh.get_review_threads.return_value = self._threads_data(
            [{"isResolved": True, "comments": {"nodes": [{}]}}]
        )
        with (
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker.tasks.complete_by_title") as mock_complete,
        ):
            _auto_complete_ask_tasks(tmp_path, gh, "owner/repo", 1)
        mock_complete.assert_not_called()


class TestSyncTasks:
    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state_with_issue(self, fido_dir: Path, issue: int = 1) -> None:
        (fido_dir / "state.json").write_text(f'{{"issue": {issue}}}')

    def test_warns_when_git_dir_not_resolved(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = MagicMock()
        with (
            patch(
                "kennel.worker._resolve_git_dir",
                side_effect=subprocess.CalledProcessError(1, "git"),
            ),
            caplog.at_level(logging.WARNING, logger="kennel"),
        ):
            sync_tasks(tmp_path, gh)
        assert "could not resolve git dir" in caplog.text

    def test_returns_early_when_no_issue_in_state(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        with patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent):
            sync_tasks(tmp_path, gh)
        gh.find_pr.assert_not_called()

    def test_returns_early_when_lock_held(self, tmp_path: Path) -> None:
        import fcntl

        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        sync_lock_path = fido_dir / "sync.lock"
        with open(sync_lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            with patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent):
                sync_tasks(tmp_path, gh)
        gh.find_pr.assert_not_called()

    def test_returns_early_when_no_open_pr(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = None
        with patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent):
            sync_tasks(tmp_path, gh)
        gh.get_pr_body.assert_not_called()

    def test_returns_early_when_pr_not_open(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "MERGED"}
        with patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent):
            sync_tasks(tmp_path, gh)
        gh.get_pr_body.assert_not_called()

    def test_returns_early_when_no_tasks(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        with (
            patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent),
            patch("kennel.worker.tasks.list_tasks", return_value=[]),
            patch("kennel.worker._auto_complete_ask_tasks"),
        ):
            sync_tasks(tmp_path, gh)
        gh.get_pr_body.assert_not_called()

    def test_syncs_pr_body_when_queue_markers_present(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        body = "desc\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->\nfooter"
        gh.get_pr_body.return_value = body
        task = {"title": "Do it", "status": "pending"}
        with (
            patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent),
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker._auto_complete_ask_tasks"),
        ):
            sync_tasks(tmp_path, gh)
        gh.edit_pr_body.assert_called_once()
        new_body = gh.edit_pr_body.call_args[0][2]
        assert "Do it" in new_body
        assert "old" not in new_body

    def test_skips_edit_when_no_queue_markers(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        gh.get_pr_body.return_value = "no markers here"
        task = {"title": "Do it", "status": "pending"}
        with (
            patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent),
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker._auto_complete_ask_tasks"),
        ):
            sync_tasks(tmp_path, gh)
        gh.edit_pr_body.assert_not_called()

    def test_get_repo_info_exception_logged(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.side_effect = RuntimeError("no remote")
        with patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent):
            sync_tasks(tmp_path, gh)
        gh.find_pr.assert_not_called()

    def test_get_pr_body_exception_breaks_loop(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        gh.get_pr_body.side_effect = RuntimeError("api down")
        task = {"title": "Do it", "status": "pending"}
        with (
            patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent),
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker._auto_complete_ask_tasks"),
        ):
            sync_tasks(tmp_path, gh)
        gh.edit_pr_body.assert_not_called()

    def test_edit_pr_body_exception_breaks_loop(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        body = "desc\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->"
        gh.get_pr_body.return_value = body
        gh.edit_pr_body.side_effect = RuntimeError("api down")
        task = {"title": "Do it", "status": "pending"}
        with (
            patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent),
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker._auto_complete_ask_tasks"),
        ):
            sync_tasks(tmp_path, gh)  # should not raise

    def test_resyncs_when_tasks_file_changes_during_sync(self, tmp_path: Path) -> None:
        gh = MagicMock()
        fido_dir = self._fido_dir(tmp_path)
        self._state_with_issue(fido_dir)
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido-bot"
        gh.find_pr.return_value = {"number": 5, "state": "OPEN"}
        body = "desc\n<!-- WORK_QUEUE_START -->\nold\n<!-- WORK_QUEUE_END -->"
        gh.get_pr_body.return_value = body
        task = {"title": "Do it", "status": "pending"}
        task_file = fido_dir / "tasks.json"
        call_count = 0

        def fake_edit(repo, pr, new_body):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate tasks.json appearing after sync started (file didn't exist before)
                task_file.write_text("[]")

        gh.edit_pr_body.side_effect = fake_edit
        # task_file does NOT exist before sync starts — mtime_before will be 0

        with (
            patch("kennel.worker._resolve_git_dir", return_value=fido_dir.parent),
            patch("kennel.worker.tasks.list_tasks", return_value=[task]),
            patch("kennel.worker._auto_complete_ask_tasks"),
        ):
            sync_tasks(tmp_path, gh)
        assert call_count == 2


class TestSyncTasksBackground:
    def test_starts_daemon_thread(self, tmp_path: Path) -> None:
        gh = MagicMock()
        started_threads = []

        def capture_start(self_t):
            started_threads.append(self_t)
            # Don't actually start the thread to avoid real sync

        with patch("threading.Thread.start", capture_start):
            sync_tasks_background(tmp_path, gh)

        assert len(started_threads) == 1
        assert started_threads[0].daemon is True

    def test_thread_name_includes_dir_name(self, tmp_path: Path) -> None:
        gh = MagicMock()
        captured = []

        def capture_start(self_t):
            captured.append(self_t)

        with patch("threading.Thread.start", capture_start):
            sync_tasks_background(tmp_path, gh)

        assert tmp_path.name in captured[0].name

    def test_thread_target_is_sync_tasks(self, tmp_path: Path) -> None:
        gh = MagicMock()
        captured = []

        def capture_start(self_t):
            captured.append(self_t)

        with patch("threading.Thread.start", capture_start):
            sync_tasks_background(tmp_path, gh)

        assert captured[0]._target is sync_tasks


class TestWorkerThread:
    def _make_thread(self, tmp_path: Path) -> WorkerThread:
        return WorkerThread(tmp_path, MagicMock())

    # ── constructor / attributes ──────────────────────────────────────────

    def test_is_daemon(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt.daemon is True

    def test_name_includes_work_dir_name(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert tmp_path.name in wt.name

    def test_work_dir_stored(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert wt.work_dir == tmp_path

    # ── wake / stop ───────────────────────────────────────────────────────

    def test_wake_sets_event(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        assert not wt._wake.is_set()
        wt.wake()
        assert wt._wake.is_set()

    def test_stop_sets_flag_and_wakes(self, tmp_path: Path) -> None:
        wt = self._make_thread(tmp_path)
        wt.stop()
        assert wt._stop is True
        assert wt._wake.is_set()

    # ── loop behaviour ────────────────────────────────────────────────────

    def _run_thread(self, wt: "WorkerThread", timeout: float = 5.0) -> None:
        """Start thread and join with timeout. Fail if it hangs."""
        wt.start()
        wt.join(timeout=timeout)
        assert not wt.is_alive(), "WorkerThread hung — did not exit within timeout"

    def test_return_1_loops_immediately_without_waiting(self, tmp_path: Path) -> None:
        """Return 1 (did work) should never call _wake.wait."""
        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake
        calls: list[int] = []

        def fake_worker_run(self_ignored=None) -> int:
            calls.append(len(calls))
            if len(calls) < 3:
                return 1
            wt._stop = True
            return 1

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        assert len(calls) == 3
        mock_wake.wait.assert_not_called()

    def test_return_0_waits_with_idle_timeout(self, tmp_path: Path) -> None:
        """Return 0 (no work) should call _wake.wait with _IDLE_TIMEOUT."""
        import kennel.worker as wmod

        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake

        def fake_worker_run(self_ignored=None) -> int:
            wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        mock_wake.wait.assert_called_once_with(timeout=wmod._IDLE_TIMEOUT)

    def test_return_2_waits_with_retry_timeout(self, tmp_path: Path) -> None:
        """Return 2 (lock contended) should call _wake.wait with _RETRY_TIMEOUT."""
        import kennel.worker as wmod

        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake

        def fake_worker_run(self_ignored=None) -> int:
            wt._stop = True
            return 2

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        mock_wake.wait.assert_called_once_with(timeout=wmod._RETRY_TIMEOUT)

    def test_exception_is_caught_and_loop_continues(self, tmp_path: Path) -> None:
        """An unexpected exception should be caught; loop continues after wait."""
        import kennel.worker as wmod

        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake
        call_count = 0

        def fake_worker_run(self_ignored=None) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        assert call_count == 2
        assert mock_wake.wait.call_count == 2
        first_timeout = mock_wake.wait.call_args_list[0].kwargs["timeout"]
        assert first_timeout == wmod._ERROR_TIMEOUT

    def test_stop_flag_exits_loop_before_next_iteration(self, tmp_path: Path) -> None:
        """Setting stop before run() starts should cause immediate exit."""
        wt = self._make_thread(tmp_path)
        wt.stop()
        with patch.object(Worker, "run") as mock_run:
            wt.run()
        mock_run.assert_not_called()

    def test_wake_clears_event_after_wait(self, tmp_path: Path) -> None:
        """_wake event is cleared after each wait so the next idle wait blocks."""
        wt = self._make_thread(tmp_path)
        mock_wake = MagicMock()
        wt._wake = mock_wake
        call_count = 0

        def fake_worker_run(self_ignored=None) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                wt._stop = True
            return 0

        with patch.object(Worker, "run", fake_worker_run):
            self._run_thread(wt)

        # clear() called after each wait
        assert mock_wake.clear.call_count == 2
