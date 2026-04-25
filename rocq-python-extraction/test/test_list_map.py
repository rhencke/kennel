from polymorphism import list_map


def test_list_map_round_trip(build_default) -> None:
    assert list_map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    assert list_map(str, [1, 2, 3]) == ["1", "2", "3"]

    source = (build_default / "polymorphism.py").read_text()
    assert "TypeVar" in source, "polymorphism.py must declare TypeVars"
    assert "def list_map" in source, "polymorphism.py must define list_map"
    assert "-> list[" in source, "polymorphism.py must annotate the return type"
