from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kennel.config import RepoConfig
from kennel.provider import ProviderID
from kennel.provider_factory import DefaultProviderFactory, extract_provider_session_id


class TestDefaultProviderFactory:
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

    def test_create_provider_rejects_unknown_provider(self, tmp_path: Path) -> None:
        system_file = tmp_path / "persona.md"
        system_file.write_text("")
        factory = DefaultProviderFactory(session_system_file=system_file)
        repo_cfg = RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            provider=ProviderID.GEMINI,
        )
        with pytest.raises(ValueError, match="unsupported provider"):
            factory.create_provider(
                repo_cfg,
                work_dir=tmp_path,
                repo_name="owner/repo",
                session=None,
            )


class TestExtractProviderSessionId:
    def test_extracts_claude_session_id(self) -> None:
        provider = SimpleNamespace(provider_id=ProviderID.CLAUDE_CODE)
        assert (
            extract_provider_session_id(
                provider,
                '{"type":"result","session_id":"claude-sess"}',
            )
            == "claude-sess"
        )

    def test_extracts_copilot_session_id(self) -> None:
        provider = SimpleNamespace(provider_id=ProviderID.COPILOT_CLI)
        assert (
            extract_provider_session_id(
                provider,
                '{"type":"result","sessionId":"copilot-sess"}',
            )
            == "copilot-sess"
        )

    def test_rejects_unknown_provider(self) -> None:
        provider = SimpleNamespace(provider_id=ProviderID.GEMINI)
        with pytest.raises(ValueError, match="unsupported provider"):
            extract_provider_session_id(provider, "")
