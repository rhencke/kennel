from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

EXPLICIT_TARGETS = {
    "test_core_terms_syntax.py": [
        "nat_add.py",
        "mk_pair_r.py",
        "zeros.py",
        "uint_val.py",
        "float_val.py",
        "str_val.py",
        "todo_val.py",
    ],
    "test_coinductives.py": [
        "repeat_tree.py",
        "tree_root_of_repeat.py",
        "zeros.py",
        "zeros_pair.py",
    ],
    "test_modules.py": ["Phase10Mod.py"],
    "test_point5.py": [
        "get_p5_v.py",
        "get_p5_w.py",
        "get_p5_x.py",
        "get_p5_y.py",
        "get_p5_z.py",
    ],
    "test_proj_pair_r.py": [
        "proj_first.py",
        "proj_second.py",
        "swap_pair_r.py",
    ],
    "test_source_maps.py": [
        "source_map_runtime_error.py",
        "source_map_runtime_error.pymap",
    ],
}


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DEFAULT = REPO_ROOT / "_build" / "default"

if BUILD_DEFAULT.is_dir():
    sys.path.insert(0, str(BUILD_DEFAULT))


@pytest.fixture
def build_default() -> Iterator[Path]:
    if not BUILD_DEFAULT.is_dir():
        pytest.skip("generated extraction artifacts are not present")
    yield BUILD_DEFAULT


def required_generated_files(path: Path) -> list[str]:
    explicit = EXPLICIT_TARGETS.get(path.name)
    if explicit is not None:
        return explicit
    return [f"{path.stem.removeprefix('test_')}.py"]


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    path = Path(str(collection_path))
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return False

    if not BUILD_DEFAULT.is_dir():
        return True

    return any(
        not (BUILD_DEFAULT / target).is_file()
        for target in required_generated_files(path)
    )
