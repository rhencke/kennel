from eq_dec_bool import Left, Right, eq_dec_bool


def test_eq_dec_bool_round_trip(build_default) -> None:
    assert eq_dec_bool(Left()) is True
    assert eq_dec_bool(Right()) is False

    source = (build_default / "eq_dec_bool.py").read_text()
    assert "return _impossible()" not in source
    assert "return __" not in source
