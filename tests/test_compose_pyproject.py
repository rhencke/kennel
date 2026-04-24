"""Tests for the ephemeral ./pyproject wrapper and composer."""

import subprocess
from pathlib import Path

import pytest

from tools.compose_pyproject import compose_fragments

REPO = Path(__file__).resolve().parents[1]


def test_compose_fragments_preserves_fragment_order(tmp_path: Path) -> None:
    first = tmp_path / "pyproject.first.toml"
    second = tmp_path / "pyproject.second.toml"
    first.write_text('[project]\nname = "fido"\n\n')
    second.write_text('[tool.ruff]\ntarget-version = "py314"\n')

    assert compose_fragments([first, second]) == (
        '[project]\nname = "fido"\n\n[tool.ruff]\ntarget-version = "py314"\n'
    )


def test_compose_fragments_rejects_duplicate_leaf_keys(tmp_path: Path) -> None:
    first = tmp_path / "pyproject.first.toml"
    second = tmp_path / "pyproject.second.toml"
    first.write_text('[project]\nname = "fido"\n')
    second.write_text('[project]\nname = "other"\n')

    with pytest.raises(ValueError, match="duplicate pyproject key 'project.name'"):
        compose_fragments([first, second])


class TestPyprojectWrapper:
    def _temp_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        (repo / "tools").mkdir(parents=True)
        (repo / "pyproject.project.toml").write_text(
            (REPO / "pyproject.project.toml").read_text()
        )
        (repo / "pyproject.build.toml").write_text(
            (REPO / "pyproject.build.toml").read_text()
        )
        (repo / "pyproject.tools.toml").write_text(
            (REPO / "pyproject.tools.toml").read_text()
        )
        (repo / "pyproject").write_text((REPO / "pyproject").read_text())
        (repo / "tools" / "compose_pyproject.py").write_text(
            (REPO / "tools" / "compose_pyproject.py").read_text()
        )
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["chmod", "+x", "pyproject"], cwd=repo, check=True)
        return repo

    def test_wrapper_creates_then_removes_rendered_file(self, tmp_path: Path) -> None:
        repo = self._temp_repo(tmp_path)

        result = subprocess.run(
            [
                "./pyproject",
                "python3",
                "-c",
                "from pathlib import Path; print(Path('pyproject.toml').is_file())",
            ],
            cwd=repo,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"
        assert not (repo / "pyproject.toml").exists()

    def test_wrapper_cleans_up_after_command_failure(self, tmp_path: Path) -> None:
        repo = self._temp_repo(tmp_path)

        result = subprocess.run(
            ["./pyproject", "sh", "-c", "exit 7"],
            cwd=repo,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 7
        assert not (repo / "pyproject.toml").exists()

    def test_wrapper_rejects_existing_rendered_file(self, tmp_path: Path) -> None:
        repo = self._temp_repo(tmp_path)
        (repo / "pyproject.toml").write_text("[project]\nname='busy'\n")

        result = subprocess.run(
            ["./pyproject", "true"],
            cwd=repo,
            check=False,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 1
        assert "pyproject.toml already exists" in result.stderr
