import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from fido.provider import ProviderID

log = logging.getLogger(__name__)


def default_sub_dir() -> Path:
    """Return the default path to the ``sub/`` skill-instructions directory.

    Single source of truth for the filesystem walk from the ``fido`` Python
    package back to the repo-root ``sub/`` tree.  The formula is
    ``parents[2]`` because this module lives at ``src/fido/config.py`` and
    ``sub/`` lives at the repo root:

    - ``parents[0]`` → ``src/fido/``
    - ``parents[1]`` → ``src/``
    - ``parents[2]`` → repo root, which contains ``sub/``

    Every module that needs this path must import this helper — do not
    redo the walk inline.  #919 and its follow-up crash were both caused
    by copies of this formula going stale during the kennel→fido rename.
    """
    return Path(__file__).resolve().parents[2] / "sub"


@dataclass(frozen=True)
class RepoMembership:
    """Cached membership info for a repo — who gets to direct fido's work.

    Populated once at server startup (``server.populate_memberships``) via
    ``GH.get_collaborators``, then shared as a field on both
    :class:`RepoConfig` (used by event dispatch at webhook time) and
    :class:`~fido.worker.RepoContext` (used by workers at task execution
    time).  Single source of truth for "who is allowed to comment on or
    approve fido's PRs on this repo".

    The bot account itself is always excluded from ``collaborators``.
    """

    collaborators: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RepoConfig:
    name: str  # "rhencke/confusio"
    work_dir: Path  # /home/rhencke/workspace/confusio
    provider: ProviderID
    membership: RepoMembership = field(default_factory=RepoMembership)


@dataclass(frozen=True)
class Config:
    port: int
    secret: bytes
    repos: dict[str, RepoConfig]  # keyed by full_name
    allowed_bots: frozenset[str]
    log_level: str
    sub_dir: Path  # path to sub/ skill files

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> Config:
        parser = argparse.ArgumentParser(
            description="fido — GitHub webhook listener for Fido"
        )
        parser.add_argument(
            "--port", type=int, default=9000, help="Listen port (default: 9000)"
        )
        parser.add_argument(
            "--secret-file",
            type=Path,
            default=Path("~/.fido-secret"),
            help="Path to webhook secret file",
        )
        parser.add_argument(
            "--allowed-bots",
            default="copilot[bot]",
            help="Comma-separated bot allowlist",
        )
        parser.add_argument(
            "--log-level",
            default="DEBUG",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        parser.add_argument(
            "repos",
            nargs="+",
            metavar="owner/repo:/path:provider",
            help="Repos to manage (name:work_dir:provider)",
        )

        args = parser.parse_args(argv)

        secret_file = args.secret_file.expanduser()
        if not secret_file.exists():
            raise SystemExit(f"secret file not found: {secret_file}")
        secret = secret_file.read_text().strip().encode()

        repos: dict[str, RepoConfig] = {}
        for spec in args.repos:
            if ":" not in spec:
                raise SystemExit(
                    f"invalid repo spec (expected name:path:provider): {spec}"
                )
            name, remainder = spec.split(":", 1)
            if ":" not in remainder:
                raise SystemExit(
                    f"invalid repo spec (expected name:path:provider): {spec}"
                )
            path_str, provider_str = remainder.rsplit(":", 1)
            try:
                provider = ProviderID(provider_str)
            except ValueError as exc:
                raise SystemExit(
                    f"invalid provider {provider_str!r} for {name}"
                ) from exc
            work_dir = Path(path_str).expanduser().resolve()
            if not work_dir.is_dir():
                raise SystemExit(f"work_dir not found: {work_dir} (for {name})")
            repos[name] = RepoConfig(name=name, work_dir=work_dir, provider=provider)

        return cls(
            port=args.port,
            secret=secret,
            repos=repos,
            allowed_bots=frozenset(
                b.strip() for b in args.allowed_bots.split(",") if b.strip()
            ),
            log_level=args.log_level.upper(),
            sub_dir=default_sub_dir(),
        )
