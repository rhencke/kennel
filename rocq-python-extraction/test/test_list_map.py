# ruff: noqa: E402
from pathlib import Path

from test_support import add_build_default_to_syspath

build_default = add_build_default_to_syspath()

from list_map import list_map


def test_list_map_round_trip() -> None:
    assert list_map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    assert list_map(str, [1, 2, 3]) == ["1", "2", "3"]

    source = (Path(build_default) / "list_map.py").read_text()
    assert "TypeVar" in source, "list_map.py must declare TypeVars"
    assert "def list_map" in source, "list_map.py must define list_map"
    assert "-> list[" in source, "list_map.py must annotate the return type"
