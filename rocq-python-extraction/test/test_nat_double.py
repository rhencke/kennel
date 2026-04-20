# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from nat_double import nat_double


def test_nat_double_round_trip() -> None:
    assert nat_double(0) == 0, "nat_double(0): got " + repr(nat_double(0))
    assert nat_double(3) == 6, "nat_double(3): got " + repr(nat_double(3))
    assert nat_double(5) == 10, "nat_double(5): got " + repr(nat_double(5))


if __name__ == "__main__":
    run_as_script(test_nat_double_round_trip, "nat round-trip: OK")
