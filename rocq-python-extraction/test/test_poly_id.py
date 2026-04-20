from poly_id import poly_id


def test_poly_id_universe_erasure(build_default) -> None:
    assert poly_id(7) == 7
    assert poly_id("woof") == "woof"

    source = (build_default / "poly_id.py").read_text()
    assert "Erased universe variables:" in source
    assert "u" in source
    assert "@{" not in source
