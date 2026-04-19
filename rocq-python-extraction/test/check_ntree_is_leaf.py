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

from ntree_is_leaf import NLeaf, NNode, NTree, ntree_is_leaf

assert ntree_is_leaf(NLeaf()) is True, "ntree_is_leaf(NLeaf()): got " + repr(
    ntree_is_leaf(NLeaf())
)
assert ntree_is_leaf(NNode([])) is False, "ntree_is_leaf(NNode([])): got " + repr(
    ntree_is_leaf(NNode([]))
)
assert ntree_is_leaf(NNode([NLeaf()])) is False, (
    "ntree_is_leaf(NNode([NLeaf()])): got " + repr(ntree_is_leaf(NNode([NLeaf()])))
)
assert isinstance(NLeaf(), NTree), "NLeaf() must be instance of NTree"
assert isinstance(NNode([]), NTree), "NNode([]) must be instance of NTree"
print("Phase 4 NTree nested inductive round-trip: OK")
