from fido.provider import ProviderID
from fido.status import _parse_repo_cmdline


class TestReposFromPid:
    def test_parses_repo_provider_from_cmdline(self, tmp_path: object) -> None:
        cmdline = f"uv\x00run\x00fido\x00owner/repo:{tmp_path}:copilot-cli\x00".encode()
        repos = _parse_repo_cmdline(cmdline)
        assert len(repos) == 1
        assert repos[0].name == "owner/repo"
        assert repos[0].provider == ProviderID.COPILOT_CLI

    def test_skips_invalid_provider_in_cmdline(self, tmp_path: object) -> None:
        cmdline = (
            f"uv\x00run\x00fido\x00owner/repo:{tmp_path}:bad-provider\x00".encode()
        )
        assert _parse_repo_cmdline(cmdline) == []

    def test_skips_repo_without_provider_in_cmdline(self, tmp_path: object) -> None:
        cmdline = f"uv\x00run\x00fido\x00owner/repo:{tmp_path}\x00".encode()
        assert _parse_repo_cmdline(cmdline) == []
