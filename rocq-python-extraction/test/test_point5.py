# ruff: noqa: E402
from test_support import add_build_default_to_syspath

add_build_default_to_syspath()

from get_p5_v import get_p5_v
from get_p5_w import get_p5_w
from get_p5_x import MkPoint5, get_p5_x
from get_p5_y import get_p5_y
from get_p5_z import get_p5_z


def test_point5_round_trip() -> None:
    p = MkPoint5(p5_x=10, p5_y=20, p5_z=30, p5_w=40, p5_v=50)

    assert get_p5_x(p) == 10, "get_p5_x: got " + repr(get_p5_x(p))
    assert get_p5_y(p) == 20, "get_p5_y: got " + repr(get_p5_y(p))
    assert get_p5_z(p) == 30, "get_p5_z: got " + repr(get_p5_z(p))
    assert get_p5_w(p) == 40, "get_p5_w: got " + repr(get_p5_w(p))
    assert get_p5_v(p) == 50, "get_p5_v: got " + repr(get_p5_v(p))

    assert p.p5_x == 10, "p.p5_x != 10"
    assert p.p5_y == 20, "p.p5_y != 20"
    assert p.p5_z == 30, "p.p5_z != 30"
    assert p.p5_w == 40, "p.p5_w != 40"
    assert p.p5_v == 50, "p.p5_v != 50"

    zero = MkPoint5(p5_x=0, p5_y=0, p5_z=0, p5_w=0, p5_v=0)
    assert get_p5_x(zero) == 0, "get_p5_x(zero)"
    assert get_p5_v(zero) == 0, "get_p5_v(zero)"

    assert isinstance(p, MkPoint5), "p must be instance of MkPoint5"
