import sys
from unittest.mock import MagicMock

import pytest


def test_main_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_main = MagicMock()
    monkeypatch.setattr("fido.main.main", mock_main)
    # Clear module cache so the import re-executes the module body.
    monkeypatch.delitem(sys.modules, "fido.__main__", raising=False)
    import fido.__main__  # noqa: F401 — importing executes main()

    mock_main.assert_called_once()
