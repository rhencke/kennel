# ruff: noqa: E402
from test_support import add_build_default_to_syspath

add_build_default_to_syspath()

from bool_not import bool_not


def test_bool_not_round_trip() -> None:
    assert bool_not(True) is False, "bool_not(True): got " + repr(bool_not(True))
    assert bool_not(False) is True, "bool_not(False): got " + repr(bool_not(False))
