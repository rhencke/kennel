from pathlib import Path

from fido.config import RepoConfig
from fido.copilotcli import CopilotCLIAPI, CopilotCLIClient
from fido.provider import ProviderID
from fido.provider_factory import DefaultProviderFactory


class TestDefaultProviderFactory:
    def test_create_api_caches_by_provider(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        repo_a = RepoConfig(
            name="owner/a",
            work_dir=tmp_path / "a",
            provider=ProviderID.CLAUDE_CODE,
        )
        repo_b = RepoConfig(
            name="owner/b",
            work_dir=tmp_path / "b",
            provider=ProviderID.CLAUDE_CODE,
        )
        assert factory.create_api(repo_a) is factory.create_api(repo_b)

    def test_create_api_builds_copilot(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        api = factory.create_api(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.COPILOT_CLI,
            )
        )
        assert api.provider_id == ProviderID.COPILOT_CLI

    def test_create_api_builds_codex(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        api = factory.create_api(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.CODEX,
            )
        )
        assert api.provider_id == ProviderID.CODEX

    def test_create_provider_builds_claude(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        provider = factory.create_provider(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.CLAUDE_CODE,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
            session=None,
        )
        assert provider.provider_id == ProviderID.CLAUDE_CODE

    def test_create_provider_builds_copilot(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        provider = factory.create_provider(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.COPILOT_CLI,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
            session=None,
        )
        assert provider.provider_id == ProviderID.COPILOT_CLI
        assert provider.agent.provider_id == ProviderID.COPILOT_CLI

    def test_create_provider_shares_copilot_api_with_create_api(
        self, tmp_path: Path
    ) -> None:
        # The same CopilotCLIAPI instance must back both the status check
        # (create_api) and the prompt path (create_provider → CopilotCLIClient).
        # If they differ, quota errors recorded on the client are invisible to
        # the pressure-pause check.
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        repo = RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            provider=ProviderID.COPILOT_CLI,
        )
        api = factory.create_api(repo)
        prov = factory.create_provider(
            repo, work_dir=tmp_path, repo_name="owner/repo", session=None
        )
        assert isinstance(api, CopilotCLIAPI)
        assert isinstance(prov.agent, CopilotCLIClient)
        assert prov.agent._quota_api is api
        assert prov.api is api

    def test_create_agent_uses_repo_provider(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        agent = factory.create_agent(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.COPILOT_CLI,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
        )
        assert agent.provider_id == ProviderID.COPILOT_CLI

    def test_create_provider_builds_codex(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        provider = factory.create_provider(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.CODEX,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
            session=None,
        )
        assert provider.provider_id == ProviderID.CODEX
        assert provider.agent.provider_id == ProviderID.CODEX


class TestProviderSessionIdExtraction:
    def test_claude_agent_extracts_session_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        agent = factory.create_agent(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.CLAUDE_CODE,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
        )
        assert (
            agent.extract_session_id('{"type":"result","session_id":"claude-sess"}')
            == "claude-sess"
        )

    def test_copilot_agent_extracts_session_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        agent = factory.create_agent(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.COPILOT_CLI,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
        )
        assert (
            agent.extract_session_id('{"type":"result","sessionId":"copilot-sess"}')
            == "copilot-sess"
        )

    def test_codex_agent_extracts_session_id(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        agent = factory.create_agent(
            RepoConfig(
                name="owner/repo",
                work_dir=tmp_path,
                provider=ProviderID.CODEX,
            ),
            work_dir=tmp_path,
            repo_name="owner/repo",
        )
        assert (
            agent.extract_session_id(
                '{"type":"thread.started","thread_id":"codex-sess"}'
            )
            == "codex-sess"
        )
