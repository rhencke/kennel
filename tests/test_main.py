"""Tests for fido.main — top-level server entry point."""

from unittest.mock import MagicMock

from fido.main import main


class TestMain:
    def test_no_args_dispatches_to_server(self) -> None:
        mock_run = MagicMock()
        main([], _server_run=mock_run)

        mock_run.assert_called_once_with()

    def test_server_args_dispatches_to_server(self) -> None:
        mock_run = MagicMock()
        main(["--port", "9000"], _server_run=mock_run)

        mock_run.assert_called_once_with()

    def test_argv_none_uses_server_path(self) -> None:
        mock_run = MagicMock()
        main(_server_run=mock_run)

        mock_run.assert_called_once_with()
