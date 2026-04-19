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

from has_left3 import Leaf, Node, Tree, has_left3

# Flat cases
assert has_left3(Leaf()) is False, "has_left3(Leaf()): got " + repr(has_left3(Leaf()))
assert has_left3(Node(1, Leaf(), Leaf())) is False, "has_left3(1-level): got " + repr(
    has_left3(Node(1, Leaf(), Leaf()))
)

# 2-level deep left child — not enough
two_deep = Node(1, Node(2, Leaf(), Leaf()), Leaf())
assert has_left3(two_deep) is False, "has_left3(2-level): got " + repr(
    has_left3(two_deep)
)

# 3-level deep left child — matches
three_deep = Node(1, Node(2, Node(3, Leaf(), Leaf()), Leaf()), Leaf())
assert has_left3(three_deep) is True, "has_left3(3-level): got " + repr(
    has_left3(three_deep)
)

# 3 levels, but going right instead of left — no match
right_deep = Node(1, Leaf(), Node(2, Node(3, Leaf(), Leaf()), Leaf()))
assert has_left3(right_deep) is False, "has_left3(right-deep): got " + repr(
    has_left3(right_deep)
)

# Type sanity
assert isinstance(Leaf(), Tree), "Leaf() must be instance of Tree"
assert isinstance(Node(0, Leaf(), Leaf()), Tree), "Node(...) must be instance of Tree"

print("Phase 5 has_left3 round-trip: OK")
