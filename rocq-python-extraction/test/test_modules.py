# ruff: noqa: E402
from test_support import add_build_default_to_syspath, run_as_script

add_build_default_to_syspath()

import Phase10Mod


def test_module_functor_round_trip() -> None:
    phase10 = Phase10Mod.Phase10Mod

    assert phase10.NatLookup.run == 0
    assert phase10.SuccLookup.run == 2
    assert phase10.NatLookup is phase10.NatLookupAgain
    assert phase10.FreshLookupA.run == 0
    assert phase10.FreshLookupB.run == 0
    assert phase10.FreshLookupA is not phase10.FreshLookupB


if __name__ == "__main__":
    run_as_script(
        test_module_functor_round_trip,
        "Module/functor round-trip: OK",
    )
