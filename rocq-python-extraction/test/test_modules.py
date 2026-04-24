import ModuleLookupFixture


def test_module_functor_round_trip() -> None:
    module_fixture = ModuleLookupFixture.ModuleLookupFixture

    assert module_fixture.NatLookup.run == 0
    assert module_fixture.SuccLookup.run == 2
    assert module_fixture.NatLookup is module_fixture.NatLookupAgain
    assert module_fixture.FreshLookupA.run == 0
    assert module_fixture.FreshLookupB.run == 0
    assert module_fixture.FreshLookupA is not module_fixture.FreshLookupB
