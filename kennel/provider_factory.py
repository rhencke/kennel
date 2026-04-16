"""Provider construction."""

from __future__ import annotations

from pathlib import Path

from kennel.claude import ClaudeClient, ClaudeCode
from kennel.config import RepoConfig
from kennel.copilotcli import CopilotCLI, CopilotCLIClient
from kennel.provider import PromptSession, Provider, ProviderAgent, ProviderID


class DefaultProviderFactory:
    """Create repo-configured provider instances."""

    def __init__(self, *, session_system_file: Path) -> None:
        self._session_system_file = session_system_file

    def create_provider(
        self,
        repo_cfg: RepoConfig,
        *,
        work_dir: Path,
        repo_name: str,
        session: PromptSession | None,
    ) -> Provider:
        match repo_cfg.provider:
            case ProviderID.CLAUDE_CODE:
                return ClaudeCode(
                    agent=ClaudeClient(
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                    )
                )
            case ProviderID.COPILOT_CLI:
                return CopilotCLI(
                    agent=CopilotCLIClient(
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                    )
                )
            case _:
                raise ValueError(f"unsupported provider: {repo_cfg.provider}")

    def create_agent(
        self,
        repo_cfg: RepoConfig,
        *,
        work_dir: Path,
        repo_name: str,
    ) -> ProviderAgent:
        return self.create_provider(
            repo_cfg,
            work_dir=work_dir,
            repo_name=repo_name,
            session=None,
        ).agent
