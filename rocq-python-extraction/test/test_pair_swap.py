from pathlib import Path

from primitives import pair_first, pair_second, pair_swap


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


def test_pair_projection_round_trip() -> None:
    assert pair_first((3, True)) == 3
    assert pair_second((3, True)) is True
    assert pair_second((7, False)) is False


def test_pair_projection_lowers_to_index_access(build_default: Path) -> None:
    source = (build_default / "primitives.py").read_text()

    assert "def fst(" not in source
    assert "def snd(" not in source
    assert "return p[0]" in source
    assert "return p[1]" in source
