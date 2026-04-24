from unittest.mock import patch


def test_main_calls_main() -> None:
    with patch("fido.main.main") as mock_main:
        import fido.__main__  # noqa: F401 — importing executes main()
    mock_main.assert_called_once()
