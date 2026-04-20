# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from roseforest_is_empty import RFCons, RFNil, RNode, roseforest_is_empty


def test_roseforest_is_empty_round_trip() -> None:
    assert roseforest_is_empty(RFNil()) is True, (
        "roseforest_is_empty(RFNil()): got " + repr(roseforest_is_empty(RFNil()))
    )
    assert roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())) is False, (
        "roseforest_is_empty(RFCons(...)): got "
        + repr(roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())))
    )


if __name__ == "__main__":
    run_as_script(test_roseforest_is_empty_round_trip, "RoseForest round-trip: OK")
