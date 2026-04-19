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

from list_add_one import list_add_one

assert list_add_one([]) == [], "list_add_one([]): got " + repr(list_add_one([]))
assert list_add_one([0, 1, 2]) == [1, 2, 3], "list_add_one([0,1,2]): got " + repr(
    list_add_one([0, 1, 2])
)
assert list_add_one([5]) == [6], "list_add_one([5]): got " + repr(list_add_one([5]))
print("Phase 3 list round-trip: OK")
