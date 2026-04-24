import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DEFAULT = REPO_ROOT / "_build" / "default"

if not BUILD_DEFAULT.is_dir():
    raise RuntimeError(
        "Rocq pytest artifacts are missing; run tests through the Docker-backed "
        "./fido tests command"
    )

sys.path.insert(0, str(BUILD_DEFAULT))


@pytest.fixture
def build_default() -> Iterator[Path]:
    yield BUILD_DEFAULT
