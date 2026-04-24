"""CLI for `fido gh-status set <message>` — set FidoCanCode's GitHub profile status."""

import logging
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from fido.claude import ClaudeClient
from fido.config import RepoConfig, default_sub_dir
from fido.github import GitHub
from fido.provider import ProviderAgent, safe_voice_turn
from fido.provider_factory import DefaultProviderFactory
from fido.status import running_repo_configs

log = logging.getLogger(__name__)

_SUB_DIR = default_sub_dir()
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

_FALLBACK_MESSAGES = (
    "Having a quiet think. Back soon.",
    "Paws tangled in the status wires for a minute.",
    "Snout in the logs. Words later.",
    "Taking a beat and wagging back soon.",
    "Brain buffering. Tail still works though.",
    "Having a small dog moment. Back in a sec.",
    "Status machine made a funny noise. Woof later.",
    "A little scrambled right now. I will be back soon.",
    "Stepping away to sniff out the right words.",
    "Temporarily unavailable, but still a very good dog.",
)


def _default_provider_factories(
    *,
    _running_repo_configs_fn: Callable[[], list[RepoConfig]] = running_repo_configs,
) -> tuple[Callable[[], ProviderAgent], ...]:
    """Return provider factories for the providers configured on the live fido."""

    repo_cfgs = _running_repo_configs_fn()
    if not repo_cfgs:
        raise RuntimeError("No running fido repo configs found")

    provider_factory = DefaultProviderFactory(session_system_file=_PERSONA_PATH)
    factories: list[Callable[[], ProviderAgent]] = []
    seen_providers: set[object] = set()
    for repo_cfg in repo_cfgs:
        if repo_cfg.provider in seen_providers:
            continue
        seen_providers.add(repo_cfg.provider)
        factories.append(
            lambda cfg=repo_cfg: provider_factory.create_agent(
                cfg,
                work_dir=cfg.work_dir,
                repo_name=cfg.name,
            )
        )
    return tuple(factories)


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
    result = safe_voice_turn(
        provider,
        f"Rewrite this status in Fido's voice: {message}",
        model=provider.voice_model,
        system_prompt=system,
        log_prefix="generate_persona_status",
    )
    if result is None:
        raise ValueError("humanify_status: run_turn returned empty")
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
    result = provider.generate_status_emoji(
        f"Pick an emoji for this status: {status_text}",
        system,
        model=provider.voice_model,
    )
    if not result:
        raise ValueError("generate_persona_emoji: generate_status_emoji returned empty")
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
    if provider is not None:
        providers = (provider,)
    else:
        provider_factories = (
            _default_provider_factories()
            if _provider_factories is None
            else tuple(_provider_factories)
        )
        providers = _candidate_providers(None, provider_factories)
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
    _gh.set_user_status(_choice(_FALLBACK_MESSAGES), ":sleeping:", busy=True)


def main(argv: list[str] | None = None, *, _GitHub: type[GitHub] = GitHub) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != "set":
        print("Usage: fido-gh-status set <message>", file=sys.stderr)
        raise SystemExit(1)
    set_gh_status(" ".join(argv[1:]), _gh=_GitHub())


if __name__ == "__main__":  # pragma: no cover
    main()
