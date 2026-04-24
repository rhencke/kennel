"""Tests for fido.main — top-level server entry point."""

from unittest.mock import patch

from fido.main import main


class TestMain:
    def test_no_args_dispatches_to_server(self) -> None:
        with patch("fido.server.run") as mock_run:
            main([])

        mock_run.assert_called_once_with()

    def test_server_args_dispatches_to_server(self) -> None:
        with patch("fido.server.run") as mock_run:
            main(["--port", "9000"])

        mock_run.assert_called_once_with()

    def test_argv_none_uses_server_path(self) -> None:
        with (
            patch("sys.argv", ["fido"]),
            patch("fido.server.run") as mock_run,
        ):
            main()

        mock_run.assert_called_once_with()
