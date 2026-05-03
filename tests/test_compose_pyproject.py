"""Tests for the ephemeral ./pyproject wrapper and composer."""

import subprocess
from pathlib import Path

import pytest

from tools.compose_pyproject import _discover_fragments, compose_fragments

REPO = Path(__file__).resolve().parents[1]


def test_discover_fragments_globs_pyproject_files(tmp_path: Path) -> None:
    (tmp_path / "pyproject.alpha.toml").write_text("[a]\nx = 1\n")
    (tmp_path / "pyproject.beta.toml").write_text("[b]\ny = 2\n")
    (tmp_path / "pyproject.toml").write_text("[c]\nz = 3\n")  # not a fragment
    (tmp_path / "ignored.toml").write_text("[d]\nw = 4\n")

    discovered = _discover_fragments(tmp_path)

    assert [path.name for path in discovered] == [
        "pyproject.alpha.toml",
        "pyproject.beta.toml",
    ]


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
        for fragment in sorted(REPO.glob("pyproject.*.toml")):
            (repo / fragment.name).write_text(fragment.read_text())
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
