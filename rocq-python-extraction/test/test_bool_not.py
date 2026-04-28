from primitives import bool_and, bool_neg, bool_not


def test_bool_not_round_trip() -> None:
    assert bool_not(True) is False, "bool_not(True): got " + repr(bool_not(True))
    assert bool_not(False) is True, "bool_not(False): got " + repr(bool_not(False))


def test_bool_and_round_trip() -> None:
    assert bool_and(True, True) is True
    assert bool_and(True, False) is False
    assert bool_and(False, True) is False
    assert bool_and(False, False) is False


def test_bool_neg_round_trip() -> None:
    assert bool_neg(True) is False
    assert bool_neg(False) is True


def test_bool_and_lowers_to_native_and(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "return b1 and b2" in source


def test_bool_neg_lowers_to_native_not(build_default) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def negb(" not in source
    assert "return not b" in source
