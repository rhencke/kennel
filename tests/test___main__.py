from __future__ import annotations

from unittest.mock import patch


def test_main_calls_main() -> None:
    with patch("kennel.main.main") as mock_main:
        import kennel.__main__  # noqa: F401 — importing executes main()
    mock_main.assert_called_once()
