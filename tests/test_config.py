from pathlib import Path

import pytest

from fido.config import Config
from fido.provider import ProviderID


class TestFromArgs:
    def test_basic_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("my-secret")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = Config.from_args(
            [
                "--secret-file",
                str(secret_file),
                f"owner/repo:{repo_dir}:claude-code",
            ]
        )
        assert cfg.secret == b"my-secret"
        assert "owner/repo" in cfg.repos
        assert cfg.repos["owner/repo"].work_dir == repo_dir
        assert cfg.repos["owner/repo"].name == "owner/repo"

    def test_defaults(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = Config.from_args(
            [
                "--secret-file",
                str(secret_file),
                f"owner/repo:{repo_dir}:claude-code",
            ]
        )
        assert cfg.port == 9000
        assert cfg.log_level == "DEBUG"
        assert "copilot[bot]" in cfg.allowed_bots

    def test_custom_port(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = Config.from_args(
            [
                "--port",
                "8080",
                "--secret-file",
                str(secret_file),
                f"owner/repo:{repo_dir}:claude-code",
            ]
        )
        assert cfg.port == 8080

    def test_custom_bots(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = Config.from_args(
            [
                "--allowed-bots",
                "bot1[bot],bot2[bot]",
                "--secret-file",
                str(secret_file),
                f"owner/repo:{repo_dir}:claude-code",
            ]
        )
        assert cfg.allowed_bots == frozenset({"bot1[bot]", "bot2[bot]"})

    def test_missing_secret_file(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        with pytest.raises(SystemExit):
            Config.from_args(
                [
                    "--secret-file",
                    str(tmp_path / "nonexistent"),
                    f"owner/repo:{repo_dir}:claude-code",
                ]
            )

    def test_invalid_repo_spec(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        with pytest.raises(SystemExit):
            Config.from_args(
                [
                    "--secret-file",
                    str(secret_file),
                    "owner-repo-no-colon",
                ]
            )

    def test_work_dir_not_found(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        with pytest.raises(SystemExit):
            Config.from_args(
                [
                    "--secret-file",
                    str(secret_file),
                    f"owner/repo:{tmp_path / 'nonexistent'}:claude-code",
                ]
            )

    def test_multiple_repos(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        dir1 = tmp_path / "repo1"
        dir1.mkdir()
        dir2 = tmp_path / "repo2"
        dir2.mkdir()
        cfg = Config.from_args(
            [
                "--secret-file",
                str(secret_file),
                f"owner/repo1:{dir1}:claude-code",
                f"owner/repo2:{dir2}:copilot-cli",
            ]
        )
        assert "owner/repo1" in cfg.repos
        assert "owner/repo2" in cfg.repos

    def test_sub_dir_points_to_package_parent(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = Config.from_args(
            [
                "--secret-file",
                str(secret_file),
                f"owner/repo:{repo_dir}:claude-code",
            ]
        )
        assert cfg.sub_dir.name == "sub"

    def test_repo_provider_parses_from_args(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = Config.from_args(
            [
                "--secret-file",
                str(secret_file),
                f"owner/repo:{repo_dir}:copilot-cli",
            ]
        )
        assert cfg.repos["owner/repo"].provider == ProviderID.COPILOT_CLI

    def test_invalid_provider_raises(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        with pytest.raises(SystemExit, match="invalid provider"):
            Config.from_args(
                [
                    "--secret-file",
                    str(secret_file),
                    f"owner/repo:{repo_dir}:wat",
                ]
            )

    def test_missing_provider_raises(self, tmp_path: Path) -> None:
        secret_file = tmp_path / "secret"
        secret_file.write_text("s")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        with pytest.raises(SystemExit, match="expected name:path:provider"):
            Config.from_args(
                [
                    "--secret-file",
                    str(secret_file),
                    f"owner/repo:{repo_dir}",
                ]
            )
