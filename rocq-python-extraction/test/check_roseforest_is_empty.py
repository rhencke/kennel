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

from roseforest_is_empty import RFCons, RFNil, RNode, roseforest_is_empty

assert roseforest_is_empty(RFNil()) is True, (
    "roseforest_is_empty(RFNil()): got " + repr(roseforest_is_empty(RFNil()))
)
assert roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())) is False, (
    "roseforest_is_empty(RFCons(...)): got "
    + repr(roseforest_is_empty(RFCons(RNode(1, RFNil()), RFNil())))
)
print("Phase 4 RoseForest round-trip: OK")
