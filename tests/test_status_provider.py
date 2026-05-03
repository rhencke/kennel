from pathlib import Path

import pytest

from fido.provider import ProviderID
from fido.status import _repos_from_pid


class TestReposFromPid:
    def test_parses_repo_provider_from_cmdline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid = 1234
        cmdline = f"uv\x00run\x00fido\x00owner/repo:{tmp_path}:copilot-cli\x00".encode()
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: cmdline if self == Path(f"/proc/{pid}/cmdline") else b"",
        )
        repos = _repos_from_pid(pid)
        assert len(repos) == 1
        assert repos[0].name == "owner/repo"
        assert repos[0].provider == ProviderID.COPILOT_CLI

    def test_skips_invalid_provider_in_cmdline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid = 1234
        cmdline = (
            f"uv\x00run\x00fido\x00owner/repo:{tmp_path}:bad-provider\x00".encode()
        )
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: cmdline if self == Path(f"/proc/{pid}/cmdline") else b"",
        )
        assert _repos_from_pid(pid) == []

    def test_skips_repo_without_provider_in_cmdline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid = 1234
        cmdline = f"uv\x00run\x00fido\x00owner/repo:{tmp_path}\x00".encode()
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda self: cmdline if self == Path(f"/proc/{pid}/cmdline") else b"",
        )
        assert _repos_from_pid(pid) == []
