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

from nat_double import nat_double

assert nat_double(0) == 0, "nat_double(0): got " + repr(nat_double(0))
assert nat_double(3) == 6, "nat_double(3): got " + repr(nat_double(3))
assert nat_double(5) == 10, "nat_double(5): got " + repr(nat_double(5))
print("Phase 3 nat round-trip: OK")
