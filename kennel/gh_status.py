"""CLI for `kennel gh-status set <message>` — set FidoCanCode's GitHub profile status."""

from __future__ import annotations

import sys
from pathlib import Path

from kennel.claude import ClaudeClient
from kennel.github import GitHub
from kennel.provider import Provider

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
    claude_client: Provider | None = None,
) -> str:
    if claude_client is None:
        claude_client = ClaudeClient()
    system = f"{persona}\n\n{_STATUS_SYSTEM}" if persona else _STATUS_SYSTEM
    result = claude_client.print_prompt(
        prompt=f"Rewrite this status in Fido's voice: {message}",
        model="claude-opus-4-6",
        system_prompt=system,
    )
    if not result:
        raise ValueError("humanify_status: print_prompt returned empty")
    return result


def generate_persona_emoji(
    status_text: str,
    persona: str,
    *,
    claude_client: Provider | None = None,
) -> str:
    if claude_client is None:
        claude_client = ClaudeClient()
    system = f"{persona}\n\n{_EMOJI_SYSTEM}" if persona else _EMOJI_SYSTEM
    result = claude_client.print_prompt_json(
        prompt=f"Pick an emoji for this status: {status_text}",
        key="emoji",
        model="claude-opus-4-6",
        system_prompt=system,
    )
    if not result:
        raise ValueError("generate_persona_emoji: print_prompt_json returned empty")
    return result


def set_gh_status(
    message: str,
    *,
    persona_path: Path = _PERSONA_PATH,
    claude_client: Provider | None = None,
    _gh: GitHub,
) -> None:
    if claude_client is None:
        claude_client = ClaudeClient()
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""
    text = generate_persona_status(message, persona, claude_client=claude_client)
    emoji = generate_persona_emoji(text, persona, claude_client=claude_client)
    _gh.set_user_status(text, emoji, busy=True)


def main(argv: list[str], *, _GitHub: type[GitHub] = GitHub) -> None:
    if len(argv) < 2 or argv[0] != "set":
        print("Usage: kennel gh-status set <message>", file=sys.stderr)
        raise SystemExit(1)
    set_gh_status(" ".join(argv[1:]), _gh=_GitHub())
