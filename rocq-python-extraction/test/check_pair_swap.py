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

from pair_swap import pair_swap

assert pair_swap((0, 1)) == (1, 0), "pair_swap((0, 1)): got " + repr(pair_swap((0, 1)))
assert pair_swap((3, 7)) == (7, 3), "pair_swap((3, 7)): got " + repr(pair_swap((3, 7)))
assert pair_swap((5, 5)) == (5, 5), "pair_swap((5, 5)): got " + repr(pair_swap((5, 5)))
print("Phase 3 prod round-trip: OK")
