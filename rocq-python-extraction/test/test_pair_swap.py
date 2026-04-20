# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from pair_swap import pair_swap


def test_pair_swap_round_trip() -> None:
    assert pair_swap((0, 1)) == (1, 0), "pair_swap((0, 1)): got " + repr(
        pair_swap((0, 1))
    )
    assert pair_swap((3, 7)) == (7, 3), "pair_swap((3, 7)): got " + repr(
        pair_swap((3, 7))
    )
    assert pair_swap((5, 5)) == (5, 5), "pair_swap((5, 5)): got " + repr(
        pair_swap((5, 5))
    )


if __name__ == "__main__":
    run_as_script(test_pair_swap_round_trip, "prod round-trip: OK")
