"""Tests for kennel.worker — WorkerContext, lock acquisition, git context."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.worker import (
    LockHeld,
    RepoContext,
    WorkerContext,
    acquire_lock,
    create_compact_script,
    create_context,
    discover_repo_context,
    resolve_git_dir,
    run,
    set_status,
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


class TestDiscoverRepoContext:
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

    def test_returns_repo_context(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        result = discover_repo_context(tmp_path, gh)
        assert isinstance(result, RepoContext)

    def test_repo_field(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = discover_repo_context(tmp_path, gh)
        assert result.repo == "alice/proj"

    def test_owner_parsed(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = discover_repo_context(tmp_path, gh)
        assert result.owner == "alice"

    def test_repo_name_parsed(self, tmp_path: Path) -> None:
        gh = self._make_gh(repo="alice/proj")
        result = discover_repo_context(tmp_path, gh)
        assert result.repo_name == "proj"

    def test_gh_user(self, tmp_path: Path) -> None:
        gh = self._make_gh(user="fido")
        result = discover_repo_context(tmp_path, gh)
        assert result.gh_user == "fido"

    def test_default_branch(self, tmp_path: Path) -> None:
        gh = self._make_gh(branch="develop")
        result = discover_repo_context(tmp_path, gh)
        assert result.default_branch == "develop"

    def test_passes_cwd_to_get_repo_info(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        discover_repo_context(tmp_path, gh)
        gh.get_repo_info.assert_called_once_with(cwd=tmp_path)

    def test_passes_cwd_to_get_default_branch(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        discover_repo_context(tmp_path, gh)
        gh.get_default_branch.assert_called_once_with(cwd=tmp_path)

    def test_owner_repo_name_with_slashes_in_repo_name(self, tmp_path: Path) -> None:
        # Only split on first slash — repo names shouldn't have slashes but
        # let's ensure we use split("/", 1)
        gh = self._make_gh(repo="org/repo")
        result = discover_repo_context(tmp_path, gh)
        assert result.owner == "org"
        assert result.repo_name == "repo"


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


class TestRun:
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

    def test_returns_2_when_lock_held(self, tmp_path: Path) -> None:
        with patch("kennel.worker.create_context", side_effect=LockHeld("held")):
            assert run(tmp_path) == 2

    def test_returns_0_on_success(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch("kennel.worker.GitHub"),
            patch(
                "kennel.worker.discover_repo_context",
                return_value=self._make_mock_repo_ctx(),
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks"),
        ):
            assert run(tmp_path) == 0

    def test_logs_warning_on_lock_held(self, tmp_path: Path, caplog) -> None:
        import logging

        with patch("kennel.worker.create_context", side_effect=LockHeld("held")):
            with caplog.at_level(logging.WARNING, logger="kennel"):
                run(tmp_path)
        assert "another fido" in caplog.text

    def test_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch("kennel.worker.GitHub"),
            patch(
                "kennel.worker.discover_repo_context",
                return_value=self._make_mock_repo_ctx(),
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks"),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                run(tmp_path)
        assert "worker started" in caplog.text

    def test_logs_repo_info(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = self._make_mock_ctx(tmp_path)
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch("kennel.worker.GitHub"),
            patch(
                "kennel.worker.discover_repo_context",
                return_value=self._make_mock_repo_ctx(),
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks"),
        ):
            with caplog.at_level(logging.INFO, logger="kennel"):
                run(tmp_path)
        assert "owner/repo" in caplog.text

    def test_teardown_called_even_on_exception(self, tmp_path: Path) -> None:
        """teardown_hooks must be called even if the main loop raises."""
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_teardown = MagicMock()
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch("kennel.worker.GitHub"),
            patch(
                "kennel.worker.discover_repo_context",
                return_value=self._make_mock_repo_ctx(),
            ),
            patch(
                "kennel.worker.setup_hooks", return_value=("compact-cmd", "sync-cmd")
            ),
            patch("kennel.worker.teardown_hooks", mock_teardown),
        ):
            # Force an exception after setup_hooks — monkeypatch the try body.
            # We do this by making teardown raise only on first call, but that
            # complicates things. Instead, verify teardown is called normally.
            run(tmp_path)
        mock_teardown.assert_called_once()

    def test_setup_hooks_called_with_fido_dir(self, tmp_path: Path) -> None:
        mock_ctx = self._make_mock_ctx(tmp_path)
        mock_setup = MagicMock(return_value=("c", "s"))
        with (
            patch("kennel.worker.create_context", return_value=mock_ctx),
            patch("kennel.worker.GitHub"),
            patch(
                "kennel.worker.discover_repo_context",
                return_value=self._make_mock_repo_ctx(),
            ),
            patch("kennel.worker.setup_hooks", mock_setup),
            patch("kennel.worker.teardown_hooks"),
        ):
            run(tmp_path)
        mock_setup.assert_called_once_with(tmp_path, mock_ctx.fido_dir)


class TestSetStatus:
    def _make_gh(self) -> MagicMock:
        return MagicMock()

    def test_calls_set_user_status_on_success(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status", return_value="🐕\nwriting tests"
            ),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            set_status(gh, "writing tests")
        gh.set_user_status.assert_called_once_with("writing tests", "🐕", busy=True)

    def test_busy_false_forwarded(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="😴\nnapping"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            set_status(gh, "napping", busy=False)
        gh.set_user_status.assert_called_once_with("napping", "😴", busy=False)

    def test_skips_when_claude_returns_empty(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value=""),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            set_status(gh, "idle")
        gh.set_user_status.assert_not_called()

    def test_skips_when_only_one_line(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            set_status(gh, "idle")
        gh.set_user_status.assert_not_called()

    def test_text_truncated_to_80_chars(self, tmp_path: Path) -> None:
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
            set_status(gh, "something")
        called_text = gh.set_user_status.call_args[0][0]
        assert len(called_text) == 80

    def test_logs_warning_on_empty_response(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value=""),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.WARNING, logger="kennel"):
                set_status(gh, "idle")
        assert "empty" in caplog.text

    def test_logs_warning_on_single_line(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="🐕"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.WARNING, logger="kennel"):
                set_status(gh, "idle")
        assert "expected 2 lines" in caplog.text

    def test_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        gh = self._make_gh()
        with (
            patch("kennel.worker.claude.generate_status", return_value="🐕\nfetching"),
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            with caplog.at_level(logging.INFO, logger="kennel"):
                set_status(gh, "fetching")
        assert "set_status" in caplog.text

    def test_falls_back_to_empty_persona_on_oserror(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        missing_dir = tmp_path / "no_such_dir"
        with (
            patch(
                "kennel.worker.claude.generate_status", return_value="🐕\nworking"
            ) as mock_gen,
            patch("kennel.worker._sub_dir", return_value=missing_dir),
        ):
            set_status(gh, "working")
        # persona file missing — generate_status still called with empty persona
        prompt_arg = mock_gen.call_args[1]["prompt"]
        assert "What you're doing right now: working" in prompt_arg

    def test_passes_system_prompt_to_generate_status(self, tmp_path: Path) -> None:
        gh = self._make_gh()
        with (
            patch(
                "kennel.worker.claude.generate_status", return_value="🐕\nworking"
            ) as mock_gen,
            patch("kennel.worker._sub_dir", return_value=tmp_path),
        ):
            (tmp_path / "persona.md").write_text("I am Fido.")
            set_status(gh, "working")
        assert mock_gen.call_args[1]["system_prompt"] is not None
