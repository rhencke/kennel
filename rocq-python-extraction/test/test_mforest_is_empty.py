# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

from mforest_is_empty import FCons, FNil, MLeaf, mforest_is_empty


def test_mforest_is_empty_round_trip() -> None:
    assert mforest_is_empty(FNil()) is True, "mforest_is_empty(FNil()): got " + repr(
        mforest_is_empty(FNil())
    )
    assert mforest_is_empty(FCons(MLeaf(1), FNil())) is False, (
        "mforest_is_empty(FCons(...)): got "
        + repr(mforest_is_empty(FCons(MLeaf(1), FNil())))
    )


if __name__ == "__main__":
    run_as_script(test_mforest_is_empty_round_trip, "MForest round-trip: OK")
