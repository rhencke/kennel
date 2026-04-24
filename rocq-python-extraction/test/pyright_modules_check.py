import ModuleLookupFixture

module_value = ModuleLookupFixture.ModuleLookupFixture
nat_lookup_result: int = module_value.NatLookup.run
succ_lookup_result: int = module_value.SuccLookup.run
same_lookup: bool = module_value.NatLookup is module_value.NatLookupAgain
different_lookup: bool = module_value.FreshLookupA is not module_value.FreshLookupB
map_missing: int = module_value.NatMap.missing

assert nat_lookup_result == 0
assert succ_lookup_result == 2
assert same_lookup
assert different_lookup
assert map_missing == 0
