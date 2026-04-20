# ruff: noqa: E402
from pathlib import Path

from test_support import add_build_default_to_syspath

build_default = add_build_default_to_syspath()

from poly_id import poly_id


def test_poly_id_universe_erasure() -> None:
    assert poly_id(7) == 7
    assert poly_id("woof") == "woof"

    source = (Path(build_default) / "poly_id.py").read_text()
    assert "Erased universe variables:" in source
    assert "u" in source
    assert "@{" not in source
