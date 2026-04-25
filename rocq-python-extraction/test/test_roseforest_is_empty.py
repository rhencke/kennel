from datatypes import RFCons, RFNil, RNode, roseforest_is_empty


def test_roseforest_is_empty_round_trip() -> None:
    assert roseforest_is_empty(RFNil()) is True, (
        "roseforest_is_empty(RFNil()): got " + repr(roseforest_is_empty(RFNil()))
    )
    assert roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())) is False, (
        "roseforest_is_empty(RFCons(...)): got "
        + repr(roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())))
    )
