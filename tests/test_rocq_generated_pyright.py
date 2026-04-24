import json
import sys
from pathlib import Path

import pytest

from fido import rocq_generated_pyright


class ExecCalled(Exception):
    def __init__(self, file: str, args: list[str]) -> None:
        self.file = file
        self.argv = args


def test_writes_pyright_config_and_execs_pyright(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated_dir = tmp_path / "generated"
    checks_dir = tmp_path / "checks"
    generated_dir.mkdir()
    checks_dir.mkdir()
    (generated_dir / "generated_module.py").write_text("value: int = 1\n")
    (checks_dir / "pyright_generated_check.py").write_text(
        "from generated_module import value\ncheck: int = value\n"
    )

    def fake_execvp(file: str, args: list[str]) -> None:
        raise ExecCalled(file, args)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generated-pyright",
            str(generated_dir),
            "--checks-dir",
            str(checks_dir),
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rocq_generated_pyright.os, "execvp", fake_execvp)

    with pytest.raises(ExecCalled) as exc_info:
        rocq_generated_pyright.main()

    copied_check = generated_dir / "pyright_generated_check.py"
    config_path = generated_dir / "pyrightconfig.json"
    assert copied_check.read_text() == (
        "from generated_module import value\ncheck: int = value\n"
    )
    assert exc_info.value.file == "pyright"
    assert exc_info.value.argv == ["pyright", "-p", str(config_path)]
    assert json.loads(config_path.read_text()) == {
        "include": ["pyright_generated_check.py"],
        "executionEnvironments": [{"root": ".", "extraPaths": ["."]}],
        "reportUnusedImport": False,
        "reportUnusedVariable": False,
        "reportUnknownLambdaType": False,
        "reportRedeclaration": False,
    }


def test_raises_if_exec_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated_dir = tmp_path / "generated"
    checks_dir = tmp_path / "checks"
    generated_dir.mkdir()
    checks_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generated-pyright",
            str(generated_dir),
            "--checks-dir",
            str(checks_dir),
        ],
    )

    def fake_execvp(file: str, args: list[str]) -> None:
        return None

    monkeypatch.setattr(rocq_generated_pyright.os, "execvp", fake_execvp)

    with pytest.raises(RuntimeError, match="pyright exec failed"):
        rocq_generated_pyright.main()
