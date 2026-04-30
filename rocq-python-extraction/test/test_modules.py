import ModuleLookupFixture


def test_module_functor_round_trip() -> None:
    module_fixture = ModuleLookupFixture.ModuleLookupFixture

    assert module_fixture.NatLookup.run == 0
    assert module_fixture.SuccLookup.run == 2
    assert module_fixture.NatLookup is module_fixture.NatLookupAgain
    assert module_fixture.FreshLookupA.run == 0
    assert module_fixture.FreshLookupB.run == 0
    assert module_fixture.FreshLookupA is not module_fixture.FreshLookupB


def test_nested_record_field_lowering_scope_round_trip() -> None:
    module_fixture = ModuleLookupFixture.ModuleLookupFixture
    scoped = module_fixture.ScopedLowering

    assert scoped.inside_left(scoped.sample) == 3
    assert module_fixture.outside_left(scoped.sample) == 3


def test_nested_record_field_lowering_scope_source(build_default) -> None:
    source = (build_default / "ModuleLookupFixture.py").read_text()

    assert "def inside_left(p):\n        return p.scoped_left" in source
    assert (
        "def outside_left(p):\n    return ModuleLookupFixture.ScopedLowering.scoped_left(p)"
        in source
    )
