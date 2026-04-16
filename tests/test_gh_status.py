"""Tests for kennel.gh_status — GitHub profile status CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.claude import ClaudeClient
from kennel.config import RepoConfig
from kennel.gh_status import (
    _default_provider_factories,
    generate_persona_emoji,
    generate_persona_status,
    main,
    set_gh_status,
)
from kennel.provider import ProviderID


def _client(**overrides: object) -> MagicMock:
    """Create a mock provider with optional attribute overrides."""
    client = MagicMock(spec=ClaudeClient)
    client.voice_model = "claude-opus-4-6"
    client.work_model = "claude-sonnet-4-6"
    client.brief_model = "claude-haiku-4-5"
    for k, v in overrides.items():
        setattr(client, k, v)
    return client


class TestGeneratePersonaStatus:
    def test_happy_path(self) -> None:
        mock_client = _client()
        mock_client.run_turn.return_value = "sniffing out a bug *tail wag*"
        result = generate_persona_status(
            "fixing a bug",
            "You are Fido",
            provider=mock_client,
        )
        assert result == "sniffing out a bug *tail wag*"

    def test_empty_response_raises(self) -> None:
        mock_client = _client()
        mock_client.run_turn.return_value = ""
        with pytest.raises(ValueError, match="humanify_status"):
            generate_persona_status("at the vet", "persona", provider=mock_client)

    def test_empty_persona(self) -> None:
        mock_client = _client()
        mock_client.run_turn.return_value = "woof"
        result = generate_persona_status("test", "", provider=mock_client)
        assert result == "woof"

    def test_creates_default_client_when_none(self) -> None:
        with patch(
            "kennel.gh_status.ClaudeClient",
            return_value=_client(),
        ) as mock_cls:
            mock_cls.return_value.run_turn.return_value = "woof"
            generate_persona_status("test", "persona")
            mock_cls.assert_called_once_with()


class TestGeneratePersonaEmoji:
    def test_happy_path(self) -> None:
        mock_client = _client()
        mock_client.generate_status_emoji.return_value = ":wrench:"
        result = generate_persona_emoji(
            "fixing bugs",
            "persona",
            provider=mock_client,
        )
        assert result == ":wrench:"

    def test_empty_response_raises(self) -> None:
        mock_client = _client()
        mock_client.generate_status_emoji.return_value = ""
        with pytest.raises(ValueError, match="generate_persona_emoji"):
            generate_persona_emoji("test", "persona", provider=mock_client)

    def test_empty_persona(self) -> None:
        mock_client = _client()
        mock_client.generate_status_emoji.return_value = ":rocket:"
        result = generate_persona_emoji("test", "", provider=mock_client)
        assert result == ":rocket:"

    def test_creates_default_client_when_none(self) -> None:
        with patch(
            "kennel.gh_status.ClaudeClient",
            return_value=_client(),
        ) as mock_cls:
            mock_cls.return_value.generate_status_emoji.return_value = ":dog:"
            generate_persona_emoji("test", "persona")
            mock_cls.assert_called_once_with()


class TestSetGhStatus:
    def test_happy_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("You are Fido")
        mock_gh = MagicMock()
        mock_client = _client()
        mock_client.run_turn.return_value = "sniffing around"
        mock_client.generate_status_emoji.return_value = ":dog2:"

        set_gh_status(
            "diagnosing issue",
            persona_path=persona_file,
            provider=mock_client,
            _gh=mock_gh,
        )
        mock_gh.set_user_status.assert_called_once_with(
            "sniffing around", ":dog2:", busy=True
        )

    def test_missing_persona_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        mock_gh = MagicMock()
        mock_client = _client()
        mock_client.run_turn.return_value = "woof"
        mock_client.generate_status_emoji.return_value = ":dog:"

        set_gh_status(
            "test",
            persona_path=tmp_path / "nonexistent.md",
            provider=mock_client,
            _gh=mock_gh,
        )
        # Verify empty persona was passed through
        call_kwargs = mock_client.run_turn.call_args
        assert call_kwargs is not None
        system = call_kwargs.kwargs.get("system_prompt", "")
        assert system.startswith("\n\n") or "rewriting a status" in system
        mock_gh.set_user_status.assert_called_once()

    def test_creates_default_client_when_none(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("persona")
        mock_gh = MagicMock()
        mock_client = _client()
        mock_client.run_turn.return_value = "woof"
        mock_client.generate_status_emoji.return_value = ":dog:"
        with patch(
            "kennel.gh_status._default_provider_factories",
            return_value=(lambda: mock_client,),
        ):
            set_gh_status("test", persona_path=persona_file, _gh=mock_gh)
        mock_gh.set_user_status.assert_called_once_with("woof", ":dog:", busy=True)

    def test_tries_next_provider_when_first_fails(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("persona")
        mock_gh = MagicMock()
        first = _client()
        first.run_turn.side_effect = RuntimeError("nope")
        second = _client()
        second.run_turn.return_value = "back soon"
        second.generate_status_emoji.return_value = ":dog:"

        set_gh_status(
            "test",
            persona_path=persona_file,
            _gh=mock_gh,
            _provider_factories=(lambda: first, lambda: second),
        )

        mock_gh.set_user_status.assert_called_once_with("back soon", ":dog:", busy=True)

    def test_falls_back_to_generic_message_when_all_providers_fail(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("persona")
        mock_gh = MagicMock()
        first = _client()
        first.run_turn.side_effect = RuntimeError("boom")
        second = _client()
        second.run_turn.side_effect = RuntimeError("still boom")
        set_gh_status(
            "test",
            persona_path=persona_file,
            _gh=mock_gh,
            _provider_factories=(lambda: first, lambda: second),
            _choice=lambda options: options[7],
        )

        mock_gh.set_user_status.assert_called_once_with(
            "A little scrambled right now. I will be back soon.",
            ":sleeping:",
            busy=True,
        )


class TestDefaultProviderFactories:
    def test_returns_no_factories_when_no_live_kennel(self) -> None:
        assert _default_provider_factories(_running_repo_configs_fn=lambda: []) == ()

    def test_uses_generic_fallback_when_no_live_kennel(self, tmp_path: Path) -> None:
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("persona")
        mock_gh = MagicMock()

        set_gh_status(
            "test",
            persona_path=persona_file,
            _gh=mock_gh,
            _provider_factories=(),
            _choice=lambda options: options[0],
        )

        mock_gh.set_user_status.assert_called_once_with(
            "Having a quiet think. Back soon.",
            ":sleeping:",
            busy=True,
        )

    def test_uses_each_configured_provider_once(self, tmp_path: Path) -> None:
        claude_cfg = RepoConfig(
            name="owner/repo-a",
            work_dir=tmp_path / "a",
            provider=ProviderID.CLAUDE_CODE,
        )
        copilot_cfg = RepoConfig(
            name="owner/repo-b",
            work_dir=tmp_path / "b",
            provider=ProviderID.COPILOT_CLI,
        )
        duplicate_claude_cfg = RepoConfig(
            name="owner/repo-c",
            work_dir=tmp_path / "c",
            provider=ProviderID.CLAUDE_CODE,
        )
        for cfg in (claude_cfg, copilot_cfg, duplicate_claude_cfg):
            cfg.work_dir.mkdir()
        factory = MagicMock()
        first = _client()
        second = _client()
        factory.create_agent.side_effect = [first, second]
        with (
            patch("kennel.gh_status.DefaultProviderFactory", return_value=factory),
        ):
            factories = _default_provider_factories(
                _running_repo_configs_fn=lambda: [
                    claude_cfg,
                    copilot_cfg,
                    duplicate_claude_cfg,
                ],
            )
            assert [build() for build in factories] == [first, second]
        factory.create_agent.assert_any_call(
            claude_cfg,
            work_dir=claude_cfg.work_dir,
            repo_name=claude_cfg.name,
        )
        factory.create_agent.assert_any_call(
            copilot_cfg,
            work_dir=copilot_cfg.work_dir,
            repo_name=copilot_cfg.name,
        )
        assert factory.create_agent.call_count == 2


class TestMain:
    def test_set_message(self) -> None:
        calls: list[str] = []

        def fake_set(msg: str, **kw: object) -> None:
            calls.append(msg)

        import kennel.gh_status

        orig = kennel.gh_status.set_gh_status
        kennel.gh_status.set_gh_status = fake_set
        try:
            main(["set", "hello", "world"], _GitHub=MagicMock)
        finally:
            kennel.gh_status.set_gh_status = orig
        assert calls == ["hello world"]

    def test_no_args(self) -> None:
        with pytest.raises(SystemExit, match="1"):
            main([])

    def test_set_no_message(self) -> None:
        with pytest.raises(SystemExit, match="1"):
            main(["set"])

    def test_unknown_subcommand(self) -> None:
        with pytest.raises(SystemExit, match="1"):
            main(["get"])
