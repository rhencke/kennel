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

    # Both inside_left (rendered while ScopedLowering's record field
    # lowering env is in scope) and outside_left (rendered after the
    # nested module returns) lower [scoped_left p] to direct attribute
    # access — the field name itself is always unqualified, so the
    # output reads identically regardless of the surrounding scope.
    assert "def inside_left(p: Scoped_pair) -> int:\n    return p.scoped_left" in source
    assert (
        "def outside_left(p: ScopedLowering.scoped_pair) -> int:\n"
        "    return p.scoped_left"
    ) in source
