"""Tests for kennel.main — top-level entry point dispatcher."""

from __future__ import annotations

from unittest.mock import patch

from kennel.main import main


class TestMain:
    def test_task_subcommand_dispatches_to_cli(self, tmp_path) -> None:
        """'kennel task ...' should delegate to the task CLI."""
        git_dir = tmp_path / ".git" / "fido"
        git_dir.mkdir(parents=True)

        with patch("kennel.tasks.add_task") as mock_add:
            main(["task", str(tmp_path), "add", "my task"])

        mock_add.assert_called_once()

    def test_no_args_dispatches_to_server(self) -> None:
        """No 'task' prefix should invoke the server."""
        with patch("kennel.server.run") as mock_run:
            main([])

        mock_run.assert_called_once()

    def test_server_args_dispatches_to_server(self) -> None:
        """Server args (no 'task' prefix) should invoke the server."""
        with patch("kennel.server.run") as mock_run:
            main(["--port", "9000"])

        mock_run.assert_called_once()

    def test_argv_none_uses_sys_argv(self) -> None:
        """When argv is None, sys.argv[1:] is used."""
        with (
            patch("sys.argv", ["kennel"]),
            patch("kennel.server.run") as mock_run,
        ):
            main()

        mock_run.assert_called_once()

    def test_argv_none_task_uses_sys_argv(self, tmp_path) -> None:
        """When argv is None and sys.argv has 'task', dispatches to CLI."""
        git_dir = tmp_path / ".git" / "fido"
        git_dir.mkdir(parents=True)

        with (
            patch("sys.argv", ["kennel", "task", str(tmp_path), "list"]),
            patch("kennel.tasks.list_tasks", return_value=[]),
        ):
            main()  # should not raise

    def test_sync_tasks_subcommand_dispatches(self, tmp_path) -> None:
        """'kennel sync-tasks <path>' should invoke sync_tasks."""
        with (
            patch("kennel.worker.sync_tasks") as mock_sync,
            patch("kennel.github.GitHub"),
        ):
            main(["sync-tasks", str(tmp_path)])
        mock_sync.assert_called_once()
        assert mock_sync.call_args[0][0] == tmp_path

    def test_sync_tasks_subcommand_defaults_to_cwd(self) -> None:
        """'kennel sync-tasks' without path uses cwd."""
        with (
            patch("kennel.worker.sync_tasks") as mock_sync,
            patch("kennel.github.GitHub"),
        ):
            main(["sync-tasks"])
        mock_sync.assert_called_once()
        from pathlib import Path

        assert mock_sync.call_args[0][0] == Path.cwd()

    def test_status_subcommand_no_repos_prints_output(self) -> None:
        """'kennel status' with no repos calls collect/format_status and prints."""
        with (
            patch("kennel.status.collect") as mock_collect,
            patch(
                "kennel.status.format_status", return_value="kennel: DOWN"
            ) as mock_fmt,
            patch("builtins.print") as mock_print,
        ):
            main(["status"])
        mock_collect.assert_called_once()
        mock_fmt.assert_called_once()
        mock_print.assert_called_once_with("kennel: DOWN")

    def test_status_subcommand_parses_repo_specs(self, tmp_path) -> None:
        """'kennel status owner/repo:/path' passes repos to collect."""
        from kennel.config import RepoConfig

        with (
            patch("kennel.status.collect") as mock_collect,
            patch("kennel.status.format_status", return_value="ok"),
            patch("builtins.print"),
        ):
            main(["status", f"owner/repo:{tmp_path}"])

        cfg = mock_collect.call_args[0][0]
        assert "owner/repo" in cfg.repos
        assert cfg.repos["owner/repo"] == RepoConfig(
            name="owner/repo", work_dir=tmp_path
        )
