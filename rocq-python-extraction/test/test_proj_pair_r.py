from records import Pair_r, pair_r_same, proj_first, proj_second, swap_pair_r


def test_proj_pair_r_round_trip() -> None:
    p1 = Pair_r(pfst_r=3, psnd_r=7)
    assert proj_first(p1) == 3, "proj_first(3, 7): got " + repr(proj_first(p1))

    p2 = Pair_r(pfst_r=0, psnd_r=99)
    assert proj_first(p2) == 0, "proj_first(0, 99): got " + repr(proj_first(p2))

    q1 = Pair_r(pfst_r=3, psnd_r=7)
    assert proj_second(q1) == 7, "proj_second(3, 7): got " + repr(proj_second(q1))

    q2 = Pair_r(pfst_r=0, psnd_r=99)
    assert proj_second(q2) == 99, "proj_second(0, 99): got " + repr(proj_second(q2))

    assert p1.pfst_r == 3, "p1.pfst_r != 3"
    assert p1.psnd_r == 7, "p1.psnd_r != 7"

    s1 = Pair_r(pfst_r=3, psnd_r=7)
    swapped = swap_pair_r(s1)
    assert swapped.pfst_r == 7, "swap(3,7).pfst_r: got " + repr(swapped.pfst_r)
    assert swapped.psnd_r == 3, "swap(3,7).psnd_r: got " + repr(swapped.psnd_r)

    double_swap = swap_pair_r(swapped)
    assert double_swap.pfst_r == s1.pfst_r, "double-swap pfst_r"
    assert double_swap.psnd_r == s1.psnd_r, "double-swap psnd_r"

    assert isinstance(p1, Pair_r), "p1 must be Pair_r"
    assert isinstance(swapped, Pair_r), "swapped must be Pair_r"

    assert pair_r_same(p1, Pair_r(pfst_r=3, psnd_r=7)) is True
    assert pair_r_same(p1, Pair_r(pfst_r=7, psnd_r=3)) is False


def test_record_fields_do_not_emit_accessor_functions(build_default) -> None:
    source = (build_default / "records.py").read_text()

    assert "def pfst_r(" not in source
    assert "def psnd_r(" not in source
    assert "return p.pfst_r" in source
    assert "return p.psnd_r" in source


def test_record_equality_lowers_to_direct_equality(build_default) -> None:
    source = (build_default / "records.py").read_text()

    assert "pair_r_eq =" not in source
    assert "__PY_NATIVE_EQ__" not in source
    assert "return left == right" in source
