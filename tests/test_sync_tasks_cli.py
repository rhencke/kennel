from pathlib import Path
from unittest.mock import MagicMock

from fido.sync_tasks_cli import main


def test_main_syncs_explicit_work_dir(tmp_path: Path) -> None:
    mock_sync = MagicMock()
    main([str(tmp_path)], _GitHub=MagicMock, _sync_tasks=mock_sync)

    mock_sync.assert_called_once()
    assert mock_sync.call_args[0][0] == tmp_path


def test_main_defaults_to_cwd() -> None:
    mock_sync = MagicMock()
    main([], _GitHub=MagicMock, _sync_tasks=mock_sync)

    mock_sync.assert_called_once()
    assert mock_sync.call_args[0][0] == Path.cwd()
