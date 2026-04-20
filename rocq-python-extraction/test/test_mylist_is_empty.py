# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from mylist_is_empty import MCons, MNil, mylist_is_empty


def test_mylist_is_empty_round_trip() -> None:
    assert mylist_is_empty(MNil()) is True, "mylist_is_empty(MNil()): got " + repr(
        mylist_is_empty(MNil())
    )
    assert mylist_is_empty(MCons(1, MNil())) is False, (
        "mylist_is_empty(MCons(...)): got " + repr(mylist_is_empty(MCons(1, MNil())))
    )


if __name__ == "__main__":
    run_as_script(test_mylist_is_empty_round_trip, "MyList round-trip: OK")
