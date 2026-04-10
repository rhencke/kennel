"""Tests for kennel.main — top-level entry point dispatcher."""

from __future__ import annotations

from unittest.mock import patch

from kennel.main import main


class TestMain:
    def test_task_subcommand_dispatches_to_cli(self, tmp_path) -> None:
        """'kennel task ...' should delegate to the task CLI."""
        git_dir = tmp_path / ".git" / "fido"
        git_dir.mkdir(parents=True)

        fake_task = {
            "id": "t1",
            "title": "my task",
            "type": "spec",
            "status": "pending",
        }
        with patch("kennel.tasks.add_task", return_value=fake_task) as mock_add:
            main(["task", str(tmp_path), "add", "spec", "my task"])

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
            patch("kennel.tasks.sync_tasks") as mock_sync,
            patch("kennel.github.GitHub"),
        ):
            main(["sync-tasks", str(tmp_path)])
        mock_sync.assert_called_once()
        assert mock_sync.call_args[0][0] == tmp_path

    def test_sync_tasks_subcommand_defaults_to_cwd(self) -> None:
        """'kennel sync-tasks' without path uses cwd."""
        with (
            patch("kennel.tasks.sync_tasks") as mock_sync,
            patch("kennel.github.GitHub"),
        ):
            main(["sync-tasks"])
        mock_sync.assert_called_once()
        from pathlib import Path

        assert mock_sync.call_args[0][0] == Path.cwd()

    def test_gh_status_subcommand_dispatches(self) -> None:
        """'kennel gh-status set msg' should delegate to gh_status.main."""
        with patch("kennel.gh_status.main") as mock_main:
            main(["gh-status", "set", "test message"])
        mock_main.assert_called_once_with(["set", "test message"])

    def test_status_subcommand_prints_output(self) -> None:
        """'kennel status' calls collect/format_status and prints the result."""
        with (
            patch("kennel.status.collect") as mock_collect,
            patch(
                "kennel.status.format_status", return_value="kennel: DOWN"
            ) as mock_fmt,
            patch("builtins.print") as mock_print,
        ):
            main(["status"])
        mock_collect.assert_called_once_with()
        mock_fmt.assert_called_once_with(mock_collect.return_value)
        mock_print.assert_called_once_with("kennel: DOWN")
