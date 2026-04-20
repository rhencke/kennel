# ruff: noqa: E402
from test_support import add_build_default_to_syspath

add_build_default_to_syspath()

from proj_first import MkPairR as MkPairR_pf
from proj_first import proj_first
from proj_second import MkPairR as MkPairR_ps
from proj_second import proj_second
from swap_pair_r import MkPairR as MkPairR_sw
from swap_pair_r import swap_pair_r


def test_proj_pair_r_round_trip() -> None:
    p1 = MkPairR_pf(pfst_r=3, psnd_r=7)
    assert proj_first(p1) == 3, "proj_first(3, 7): got " + repr(proj_first(p1))

    p2 = MkPairR_pf(pfst_r=0, psnd_r=99)
    assert proj_first(p2) == 0, "proj_first(0, 99): got " + repr(proj_first(p2))

    q1 = MkPairR_ps(pfst_r=3, psnd_r=7)
    assert proj_second(q1) == 7, "proj_second(3, 7): got " + repr(proj_second(q1))

    q2 = MkPairR_ps(pfst_r=0, psnd_r=99)
    assert proj_second(q2) == 99, "proj_second(0, 99): got " + repr(proj_second(q2))

    assert p1.pfst_r == 3, "p1.pfst_r != 3"
    assert p1.psnd_r == 7, "p1.psnd_r != 7"

    s1 = MkPairR_sw(pfst_r=3, psnd_r=7)
    swapped = swap_pair_r(s1)
    assert swapped.pfst_r == 7, "swap(3,7).pfst_r: got " + repr(swapped.pfst_r)
    assert swapped.psnd_r == 3, "swap(3,7).psnd_r: got " + repr(swapped.psnd_r)

    double_swap = swap_pair_r(swapped)
    assert double_swap.pfst_r == s1.pfst_r, "double-swap pfst_r"
    assert double_swap.psnd_r == s1.psnd_r, "double-swap psnd_r"

    assert isinstance(p1, MkPairR_pf), "p1 must be MkPairR"
    assert isinstance(swapped, MkPairR_sw), "swapped must be MkPairR"
