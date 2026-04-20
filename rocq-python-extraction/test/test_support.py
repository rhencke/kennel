"""Shared helpers for extraction acceptance checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_build_default_to_syspath() -> Path:
    candidates: list[Path] = []
    for start in (Path(os.path.abspath(__file__)).parent, Path.cwd()):
        current = start
        while True:
            if current.name == "default" and current.parent.name == "_build":
                sys.path.insert(0, str(current))
                return current
            candidate = current / "_build" / "default"
            if candidate.is_dir() and candidate not in candidates:
                candidates.append(candidate)
            if current.parent == current:
                break
            current = current.parent
    if candidates:
        build_default = candidates[-1]
        sys.path.insert(0, str(build_default))
        return build_default
    try:
        import pytest

        pytest.skip(
            "generated extraction artifacts are not present", allow_module_level=True
        )
    except ImportError as exc:
        raise RuntimeError("generated extraction artifacts are not present") from exc
