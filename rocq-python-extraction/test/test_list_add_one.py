# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from list_add_one import list_add_one


def test_list_add_one_round_trip() -> None:
    assert list_add_one([]) == [], "list_add_one([]): got " + repr(list_add_one([]))
    assert list_add_one([0, 1, 2]) == [1, 2, 3], "list_add_one([0,1,2]): got " + repr(
        list_add_one([0, 1, 2])
    )
    assert list_add_one([5]) == [6], "list_add_one([5]): got " + repr(list_add_one([5]))


if __name__ == "__main__":
    run_as_script(test_list_add_one_round_trip, "list round-trip: OK")
