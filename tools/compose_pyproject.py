#!/usr/bin/env python3
"""Compose an ephemeral pyproject.toml from repo fragments."""

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _discover_fragments(root: Path = ROOT) -> list[Path]:
    """Return ``pyproject.*.toml`` fragments present in ``root``.

    Fragments are intentionally split across files so each Dockerfile stage can
    COPY only what it needs — e.g. the typecheck stage skips
    ``pyproject.ruff.toml`` so editing ruff config does not invalidate
    pyright's cache. The glob means new fragments are picked up automatically
    and missing-but-irrelevant fragments do not break composition.
    """
    return sorted(root.glob("pyproject.*.toml"))


def _collect_leaf_paths(
    value: object, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    if isinstance(value, dict):
        paths: set[tuple[str, ...]] = set()
        for key, child in value.items():
            paths.update(_collect_leaf_paths(child, prefix + (str(key),)))
        return paths
    return {prefix}


def compose_fragments(fragments: list[Path]) -> str:
    seen: dict[tuple[str, ...], Path] = {}
    parts: list[str] = []
    for fragment in fragments:
        if not fragment.is_file():
            raise FileNotFoundError(f"missing pyproject fragment: {fragment}")
        text = fragment.read_text()
        data = tomllib.loads(text)
        for path in sorted(_collect_leaf_paths(data)):
            if path in seen:
                dotted = ".".join(path)
                raise ValueError(
                    f"duplicate pyproject key {dotted!r} in {fragment} and {seen[path]}"
                )
            seen[path] = fragment
        parts.append(text.rstrip() + "\n")
    return "\n".join(parts)


def compose() -> str:
    return compose_fragments(_discover_fragments())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "pyproject.toml")
    args = parser.parse_args()
    try:
        rendered = compose()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
