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

from bintree_is_leaf import BLeaf, BNode, bintree_is_leaf

assert bintree_is_leaf(BLeaf()) is True, "bintree_is_leaf(BLeaf()): got " + repr(
    bintree_is_leaf(BLeaf())
)
assert bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())) is False, (
    "bintree_is_leaf(BNode(...)): got "
    + repr(bintree_is_leaf(BNode(BLeaf(), 42, BLeaf())))
)
print("Phase 4 BinTree round-trip: OK")
