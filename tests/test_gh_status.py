"""Tests for kennel.gh_status — GitHub profile status CLI."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kennel.claude import ClaudeError
from kennel.gh_status import (
    generate_persona_emoji,
    generate_persona_status,
    main,
    set_gh_status,
)


class TestGeneratePersonaStatus:
    def test_happy_path(self) -> None:
        result = generate_persona_status(
            "fixing a bug",
            "You are Fido",
            _print_prompt=lambda **kw: "sniffing out a bug *tail wag*",
        )
        assert result == "sniffing out a bug *tail wag*"

    def test_empty_response_falls_back_to_raw(self) -> None:
        result = generate_persona_status(
            "at the vet", "persona", _print_prompt=lambda **kw: ""
        )
        assert result == "at the vet"

    def test_claude_error_falls_back_to_raw(self) -> None:
        def failing(**kw):
            raise ClaudeError("fail")

        result = generate_persona_status("at the vet", "persona", _print_prompt=failing)
        assert result == "at the vet"

    def test_long_message_truncated_on_fallback(self) -> None:
        long_msg = "x" * 100
        result = generate_persona_status(long_msg, "", _print_prompt=lambda **kw: "")
        assert len(result) == 80

    def test_empty_persona(self) -> None:
        result = generate_persona_status("test", "", _print_prompt=lambda **kw: "woof")
        assert result == "woof"


class TestGeneratePersonaEmoji:
    def test_happy_path(self) -> None:
        result = generate_persona_emoji(
            "fixing bugs",
            "persona",
            _print_prompt_json=lambda **kw: ":wrench:",
        )
        assert result == ":wrench:"

    def test_empty_response_falls_back_to_dog(self) -> None:
        result = generate_persona_emoji(
            "test", "persona", _print_prompt_json=lambda **kw: ""
        )
        assert result == ":dog:"

    def test_claude_error_falls_back_to_dog(self) -> None:
        def failing(**kw):
            raise ClaudeError("fail")

        result = generate_persona_emoji("test", "persona", _print_prompt_json=failing)
        assert result == ":dog:"

    def test_empty_persona(self) -> None:
        result = generate_persona_emoji(
            "test", "", _print_prompt_json=lambda **kw: ":rocket:"
        )
        assert result == ":rocket:"


class TestSetGhStatus:
    def test_happy_path(self, tmp_path) -> None:
        persona_file = tmp_path / "persona.md"
        persona_file.write_text("You are Fido")
        mock_gh = MagicMock()

        set_gh_status(
            "diagnosing issue",
            persona_path=persona_file,
            _generate_persona_status=lambda msg, p: "sniffing around",
            _generate_persona_emoji=lambda txt, p: ":dog2:",
            _get_github=lambda: mock_gh,
        )
        mock_gh.set_user_status.assert_called_once_with(
            "sniffing around", ":dog2:", busy=True
        )

    def test_missing_persona_file(self, tmp_path) -> None:
        mock_gh = MagicMock()
        calls: list[str] = []

        def track_status(msg: str, persona: str) -> str:
            calls.append(persona)
            return "woof"

        set_gh_status(
            "test",
            persona_path=tmp_path / "nonexistent.md",
            _generate_persona_status=track_status,
            _generate_persona_emoji=lambda txt, p: ":dog:",
            _get_github=lambda: mock_gh,
        )
        assert calls == [""]
        mock_gh.set_user_status.assert_called_once()


class TestMain:
    def test_set_message(self) -> None:
        calls: list[str] = []

        def fake_set(msg: str, **kw: object) -> None:
            calls.append(msg)

        import kennel.gh_status

        orig = kennel.gh_status.set_gh_status
        kennel.gh_status.set_gh_status = fake_set
        try:
            main(["set", "hello", "world"])
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
