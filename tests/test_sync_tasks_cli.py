from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fido.sync_tasks_cli import main


def test_main_syncs_explicit_work_dir(tmp_path: Path) -> None:
    sync_calls: list[tuple[Any, Any]] = []

    def fake_sync(work_dir: Path, gh: object) -> None:
        sync_calls.append((work_dir, gh))

    main([str(tmp_path)], _GitHub=MagicMock, _sync_tasks=fake_sync)

    assert len(sync_calls) == 1
    assert sync_calls[0][0] == tmp_path


def test_main_defaults_to_cwd() -> None:
    sync_calls: list[tuple[Any, Any]] = []

    def fake_sync(work_dir: Path, gh: object) -> None:
        sync_calls.append((work_dir, gh))

    main([], _GitHub=MagicMock, _sync_tasks=fake_sync)

    assert len(sync_calls) == 1
    assert sync_calls[0][0] == Path.cwd()
