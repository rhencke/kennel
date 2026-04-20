# ruff: noqa: E402
from test_support import add_build_default_to_syspath

add_build_default_to_syspath()

from option_inc import option_inc


def test_option_inc_round_trip() -> None:
    assert option_inc(None) is None, "option_inc(None): got " + repr(option_inc(None))
    assert option_inc(0) == 1, "option_inc(0): got " + repr(option_inc(0))
    assert option_inc(5) == 6, "option_inc(5): got " + repr(option_inc(5))
