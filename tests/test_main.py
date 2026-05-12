"""Tests for fido.main — top-level server entry point."""

from unittest.mock import MagicMock

import pytest

from fido.main import main


class TestMain:
    def test_no_args_dispatches_to_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = MagicMock()
        monkeypatch.setattr("fido.server.run", mock_run)
        main([])

        mock_run.assert_called_once_with()

    def test_server_args_dispatches_to_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = MagicMock()
        monkeypatch.setattr("fido.server.run", mock_run)
        main(["--port", "9000"])

        mock_run.assert_called_once_with()

    def test_argv_none_uses_server_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = MagicMock()
        monkeypatch.setattr("fido.server.run", mock_run)
        main()

        mock_run.assert_called_once_with()
