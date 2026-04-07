from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("kennel")


@dataclass(frozen=True)
class RepoConfig:
    name: str  # "rhencke/confusio"
    work_dir: Path  # /home/rhencke/workspace/confusio


@dataclass(frozen=True)
class Config:
    port: int
    secret: bytes
    repos: dict[str, RepoConfig]  # keyed by full_name
    allowed_bots: frozenset[str]
    log_level: str
    self_repo: str | None  # which repo is kennel itself (for self-restart)
    sub_dir: Path  # path to sub/ skill files

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> Config:
        parser = argparse.ArgumentParser(
            description="kennel — GitHub webhook listener for Fido"
        )
        parser.add_argument(
            "--port", type=int, default=9000, help="Listen port (default: 9000)"
        )
        parser.add_argument(
            "--secret-file",
            type=Path,
            default=Path("~/.kennel-secret"),
            help="Path to webhook secret file",
        )
        parser.add_argument(
            "--allowed-bots",
            default="copilot[bot]",
            help="Comma-separated bot allowlist",
        )
        parser.add_argument(
            "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
        )
        parser.add_argument(
            "--self-repo",
            default=None,
            help="Repo name that is kennel itself (for self-restart on merge)",
        )
        parser.add_argument(
            "repos",
            nargs="+",
            metavar="owner/repo:/path",
            help="Repos to manage (name:work_dir)",
        )

        args = parser.parse_args(argv)

        secret_file = args.secret_file.expanduser()
        if not secret_file.exists():
            raise SystemExit(f"secret file not found: {secret_file}")
        secret = secret_file.read_text().strip().encode()

        repos: dict[str, RepoConfig] = {}
        for spec in args.repos:
            if ":" not in spec:
                raise SystemExit(f"invalid repo spec (expected name:path): {spec}")
            name, path_str = spec.split(":", 1)
            work_dir = Path(path_str).expanduser().resolve()
            if not work_dir.is_dir():
                raise SystemExit(f"work_dir not found: {work_dir} (for {name})")
            repos[name] = RepoConfig(name=name, work_dir=work_dir)

        return cls(
            port=args.port,
            secret=secret,
            repos=repos,
            allowed_bots=frozenset(
                b.strip() for b in args.allowed_bots.split(",") if b.strip()
            ),
            log_level=args.log_level.upper(),
            self_repo=args.self_repo,
            sub_dir=Path(__file__).resolve().parent.parent / "sub",
        )
