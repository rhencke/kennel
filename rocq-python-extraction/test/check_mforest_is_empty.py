# ruff: noqa: E402
import os
import sys

# The extracted .py files always land in the dune workspace build root (_build/default/).
# Walk up from __file__ to find it — works whether or not dune-workspace is present.
_d = os.path.dirname(os.path.abspath(__file__))
while not (
    os.path.basename(_d) == "default"
    and os.path.basename(os.path.dirname(_d)) == "_build"
):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
del _d

from mforest_is_empty import FCons, FNil, MLeaf, mforest_is_empty

assert mforest_is_empty(FNil()) is True, "mforest_is_empty(FNil()): got " + repr(
    mforest_is_empty(FNil())
)
assert mforest_is_empty(FCons(MLeaf(1), FNil())) is False, (
    "mforest_is_empty(FCons(...)): got "
    + repr(mforest_is_empty(FCons(MLeaf(1), FNil())))
)
print("Phase 4 MForest round-trip: OK")
