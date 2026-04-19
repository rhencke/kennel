# ruff: noqa: F821
exec(open("roseforest_is_empty.py").read())
assert roseforest_is_empty(RFNil()) is True, (
    "roseforest_is_empty(RFNil()): got " + repr(roseforest_is_empty(RFNil()))
)
assert roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())) is False, (
    "roseforest_is_empty(RFCons(...)): got "
    + repr(roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())))
)
print("Phase 4 RoseForest round-trip: OK")
