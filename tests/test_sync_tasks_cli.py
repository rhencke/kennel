from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from kennel.sync_tasks_cli import main


def test_main_syncs_explicit_work_dir(tmp_path: Path) -> None:
    with patch("kennel.tasks.sync_tasks") as mock_sync:
        main([str(tmp_path)], _GitHub=MagicMock)

    mock_sync.assert_called_once()
    assert mock_sync.call_args[0][0] == tmp_path


def test_main_defaults_to_cwd() -> None:
    with patch("kennel.tasks.sync_tasks") as mock_sync:
        main([], _GitHub=MagicMock)

    mock_sync.assert_called_once()
    assert mock_sync.call_args[0][0] == Path.cwd()
