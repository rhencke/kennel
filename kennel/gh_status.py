"""CLI for `kennel gh-status set <message>` — set FidoCanCode's GitHub profile status."""

from __future__ import annotations

import logging
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from kennel.claude import ClaudeClient
from kennel.github import GitHub
from kennel.provider import ProviderAgent

log = logging.getLogger(__name__)

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

_NAP_MESSAGES = (
    "Taking a little nap. Back soon.",
    "Curled up for a quick nap.",
    "Dozing by the keyboard for a bit.",
    "Snoring softly under the desk.",
    "Resting my paws and rebooting my brain.",
    "Power nap time. Woof soon.",
    "Taking a breather in the sunbeam.",
    "Sleeping off a big debugging session.",
    "Paws up. Eyes closed. Back later.",
    "Nap mode engaged.",
    "Catching a quick snooze between bugs.",
    "Tucked in for a tiny dog nap.",
    "Dreaming about clean test runs.",
    "Resting my little programmer paws.",
    "Sleeping until the next good idea arrives.",
    "Having a cozy floor nap.",
    "Dozing while the thoughts settle.",
    "Taking five in a warm patch of sun.",
    "Sniffed too many logs. Need a nap.",
    "Recharging with a small sleepy woof.",
    "Out for a brief nap walk in dreamland.",
    "Napping now. Back when my tail reboots.",
    "Having a quiet nap and a slower heartbeat.",
    "Resting after a long code chase.",
    "Snoozing with one ear on the build.",
    "Paused for a gentle nap.",
    "Sleeping through the noisy part.",
    "Just me, a blanket, and a quick nap.",
    "Settled in for some quality dozing.",
    "Nap first. Clever fix later.",
    "Letting my brain fetch a little rest.",
    "Tiny nap. Big comeback.",
    "Taking a sleepy lap around dreamland.",
    "Tail still. Nose tucked. Napping.",
    "Having a soft little reset nap.",
    "Camping out in nap mode.",
    "Do not disturb. Dreaming of green checks.",
    "Sleeping on the next move for a minute.",
    "Back soon. Currently very snoozy.",
    "Stepping away for a nap and a stretch.",
    "Quiet paws. Closed eyes. Gentle nap.",
    "Snoozing until the next bark-worthy task.",
    "Taking a warm, lazy nap.",
    "Letting the zoomies cool off.",
    "Having a thoughtful nap about architecture.",
    "Paused for a pillow-level refactor.",
    "Napping with extreme professionalism.",
    "Resting before the next fetch.",
    "Low-power dog mode for a minute.",
    "Sleeping like a very tired code hound.",
    "Tuning out for a tiny nap.",
    "Stashed my paws and took a nap.",
    "Having a hallway nap and a nice dream.",
    "Short snooze. Long wag later.",
    "Listening to the hum of a nap.",
    "Taking a calm little timeout nap.",
    "All tucked in with my thoughts.",
    "Gone to fetch some rest.",
    "Sleeping through the lint in my dreams.",
    "Resting until the ideas line back up.",
    "A nap is in progress.",
    "Having a low-noise, high-comfort nap.",
    "Snoozing near the charging cable.",
    "Taking a reset break with a blanket.",
    "Paws folded. Brain idling. Napping.",
    "On a tiny nap detour.",
    "Resting up for the next round of fixes.",
    "Closed my laptop. Opened my nap.",
    "Having a code hound catnap.",
    "Dreaming in tidy little commits.",
    "Nap vibes only right now.",
    "Took my thoughts to bed for a bit.",
    "Sleeping off some stack traces.",
    "A short nap should do the trick.",
    "Letting the brain cache cool down.",
    "Snuggled into a productive nap.",
    "Dozing until the next tail wag.",
    "Resting in a very responsible way.",
    "Nap break. Nothing dramatic.",
    "Laying low and sleeping lightly.",
    "Having a peaceful puppy pause.",
    "Sleeping until the code smells fresher.",
    "Quietly recharging in nap mode.",
    "A comfy nap is currently deployed.",
    "Flat on the rug. Out for a bit.",
    "Taking a soft reboot nap.",
    "Pacing stopped. Napping started.",
    "Resting with excellent blanket coverage.",
    "Nose tucked, tail still, brain rebooting.",
    "Just a good old-fashioned nap.",
    "Borrowing a little rest from the afternoon.",
    "Snoozing where the light is nicest.",
    "Nap in progress. Bark later.",
    "On break with a sleepy sigh.",
    "Dreaming about elegant diffs.",
    "Having a paws-off minute.",
    "Settling into a strategic snooze.",
    "Resting the snout and the neurons.",
    "Taking the kind of nap that fixes everything.",
    "Sleeping now. Will wag again soon.",
    "One small nap for dog. One big reset for brain.",
    "Curled up and temporarily unavailable.",
)


def _default_provider_factories() -> tuple[Callable[[], ProviderAgent], ...]:
    """Return the currently available provider constructors for gh-status."""
    return (ClaudeClient,)


def _candidate_providers(
    provider: ProviderAgent | None,
    provider_factories: Sequence[Callable[[], ProviderAgent]],
) -> tuple[ProviderAgent, ...]:
    if provider is not None:
        return (provider,)
    return tuple(factory() for factory in provider_factories)


def generate_persona_status(
    message: str,
    persona: str,
    *,
    provider: ProviderAgent | None = None,
) -> str:
    if provider is None:
        provider = ClaudeClient()
    system = f"{persona}\n\n{_STATUS_SYSTEM}" if persona else _STATUS_SYSTEM
    result = provider.print_prompt(
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
    provider: ProviderAgent | None = None,
) -> str:
    if provider is None:
        provider = ClaudeClient()
    system = f"{persona}\n\n{_EMOJI_SYSTEM}" if persona else _EMOJI_SYSTEM
    result = provider.print_prompt_json(
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
    provider: ProviderAgent | None = None,
    _gh: GitHub,
    _provider_factories: Sequence[Callable[[], ProviderAgent]] | None = None,
    _choice: Callable[[Sequence[str]], str] = random.choice,
) -> None:
    try:
        persona = persona_path.read_text()
    except FileNotFoundError:
        persona = ""
    providers = _candidate_providers(
        provider,
        _default_provider_factories()
        if _provider_factories is None
        else tuple(_provider_factories),
    )
    for current_provider in providers:
        try:
            text = generate_persona_status(message, persona, provider=current_provider)
            emoji = generate_persona_emoji(text, persona, provider=current_provider)
        except Exception as exc:
            log.warning(
                "set_gh_status: provider %s failed, trying next: %s",
                getattr(current_provider, "provider_id", "unknown"),
                exc,
            )
            continue
        _gh.set_user_status(text, emoji, busy=True)
        return
    _gh.set_user_status(_choice(_NAP_MESSAGES), ":sleeping:", busy=True)


def main(argv: list[str], *, _GitHub: type[GitHub] = GitHub) -> None:
    if len(argv) < 2 or argv[0] != "set":
        print("Usage: kennel gh-status set <message>", file=sys.stderr)
        raise SystemExit(1)
    set_gh_status(" ".join(argv[1:]), _gh=_GitHub())
