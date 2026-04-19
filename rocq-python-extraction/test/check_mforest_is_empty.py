# ruff: noqa: F821
exec(open("mforest_is_empty.py").read())
assert mforest_is_empty(FNil()) is True, "mforest_is_empty(FNil()): got " + repr(
    mforest_is_empty(FNil())
)
assert mforest_is_empty(FCons(MLeaf(1), FNil())) is False, (
    "mforest_is_empty(FCons(...)): got "
    + repr(mforest_is_empty(FCons(MLeaf(1), FNil())))
)
print("Phase 4 MForest round-trip: OK")
