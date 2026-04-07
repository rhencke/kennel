"""Tests for kennel.main — top-level entry point dispatcher."""

from __future__ import annotations

from unittest.mock import patch

from kennel.main import main


class TestMain:
    def test_task_subcommand_dispatches_to_cli(self, tmp_path) -> None:
        """'kennel task ...' should delegate to the task CLI."""
        git_dir = tmp_path / ".git" / "fido"
        git_dir.mkdir(parents=True)

        with patch("kennel.cli.tasks.add_task") as mock_add:
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
            patch("kennel.cli.tasks.list_tasks", return_value=[]),
        ):
            main()  # should not raise
