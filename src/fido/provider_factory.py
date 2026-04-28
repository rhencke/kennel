"""Provider construction."""

import threading
from pathlib import Path

from fido.claude import ClaudeAPI, ClaudeClient, ClaudeCode
from fido.config import RepoConfig
from fido.copilotcli import CopilotCLI, CopilotCLIAPI, CopilotCLIClient
from fido.provider import (
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
)


class DefaultProviderFactory:
    """Create repo-configured provider instances."""

    def __init__(self, *, session_system_file: Path) -> None:
        self._session_system_file = session_system_file
        self._api_lock = threading.Lock()
        self._apis: dict[ProviderID, ProviderAPI] = {}

    def create_api(self, repo_cfg: RepoConfig) -> ProviderAPI:
        with self._api_lock:
            api = self._apis.get(repo_cfg.provider)
            if api is not None:
                return api
            match repo_cfg.provider:
                case ProviderID.CLAUDE_CODE:
                    api = ClaudeAPI()
                case ProviderID.COPILOT_CLI:
                    api = CopilotCLIAPI()
                case ProviderID.CODEX:
                    raise NotImplementedError("codex provider not yet wired")
                case _:
                    raise ValueError(f"unsupported provider: {repo_cfg.provider}")
            self._apis[repo_cfg.provider] = api
            return api

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
            case ProviderID.CODEX:
                raise NotImplementedError("codex provider not yet wired")
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
