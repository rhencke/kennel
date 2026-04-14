"""CLI for `kennel gh-status set <message>` — set FidoCanCode's GitHub profile status."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from kennel import claude
from kennel.github import GitHub

_SUB_DIR = Path(__file__).resolve().parent.parent / "sub"
_PERSONA_PATH = _SUB_DIR / "persona.md"

_STATUS_SYSTEM = (
    "You are rewriting a status message in Fido's voice (a dog who codes)."
    " Keep it under 80 characters. Output ONLY the rewritten status text,"
    " nothing else."
)

_EMOJI_SYSTEM = (
    "Pick a single GitHub emoji shortcode (like :dog: or :wrench:) that fits"
    " the status message. Output ONLY the emoji shortcode, nothing else."
)


def generate_persona_status(
    message: str,
    persona: str,
    *,
    _print_prompt: Callable[..., str] = claude.print_prompt,
) -> str:
    system = f"{persona}\n\n{_STATUS_SYSTEM}" if persona else _STATUS_SYSTEM
    result = _print_prompt(
        prompt=f"Rewrite this status in Fido's voice: {message}",
        model="claude-opus-4-6",
        system_prompt=system,
    )
    return result if result else message[:80]


def generate_persona_emoji(
    status_text: str,
    persona: str,
    *,
    _print_prompt_json: Callable[..., str] = claude.print_prompt_json,
) -> str:
    system = f"{persona}\n\n{_EMOJI_SYSTEM}" if persona else _EMOJI_SYSTEM
    result = _print_prompt_json(
        prompt=f"Pick an emoji for this status: {status_text}",
        key="emoji",
        model="claude-opus-4-6",
        system_prompt=system,
    )
    return result if result else ":dog:"


def set_gh_status(
    message: str,
    *,
    persona_path: Path = _PERSONA_PATH,
    _generate_persona_status: Callable[..., str] = generate_persona_status,
    _generate_persona_emoji: Callable[..., str] = generate_persona_emoji,
    _gh: GitHub,
) -> None:
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""
    text = _generate_persona_status(message, persona)
    emoji = _generate_persona_emoji(text, persona)
    _gh.set_user_status(text, emoji, busy=True)


def main(argv: list[str], *, _GitHub=GitHub) -> None:
    if len(argv) < 2 or argv[0] != "set":
        print("Usage: kennel gh-status set <message>", file=sys.stderr)
        raise SystemExit(1)
    set_gh_status(" ".join(argv[1:]), _gh=_GitHub())
