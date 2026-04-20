# ruff: noqa: E402
from test_support import add_build_default_to_syspath

add_build_default_to_syspath()

from mylist_is_empty import MCons, MNil, mylist_is_empty


def test_mylist_is_empty_round_trip() -> None:
    assert mylist_is_empty(MNil()) is True, "mylist_is_empty(MNil()): got " + repr(
        mylist_is_empty(MNil())
    )
    assert mylist_is_empty(MCons(1, MNil())) is False, (
        "mylist_is_empty(MCons(...)): got " + repr(mylist_is_empty(MCons(1, MNil())))
    )
